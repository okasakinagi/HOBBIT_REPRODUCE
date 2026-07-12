"""
bench_gsm8k.py — GSM8K 数学推理精度评测
=========================================
在 HOBBIT 缝合版 Mixtral 上跑 GSM8K 数学推理评测。
支持增量保存 + 断点续跑，适合 nohup 后台长时间运行。

用法（服务器 nohup 推荐）：
    # 基线模式
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b nohup python bench_gsm8k.py \
        --mode baseline > ../logs/gsm8k_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    # HOBBIT 模式
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b nohup python bench_gsm8k.py \
        --mode hobbit > ../logs/gsm8k_hobbit_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    # 排队串行（写 shell 脚本 + nohup）
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b nohup bash -c '
        GSM8K_MAX_QUESTIONS=20 python bench_gsm8k.py --mode baseline
        GSM8K_MAX_QUESTIONS=20 python bench_gsm8k.py --mode hobbit
    ' > ../logs/gsm8k_queued_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    # 断点续跑（如果 checkpoint 存在）
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b nohup python bench_gsm8k.py \
        --mode baseline --resume > ../logs/gsm8k_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &

可选参数：
    --mode baseline|hobbit  评测模式（必选，不支持 compare）
    --resume                从 checkpoint 续跑（跳过已完成的题目）
    GSM8K_MAX_QUESTIONS=50  测试题数（默认 20，全量 1319）
    GSM8K_QUESTION_START=0  起始序号

环境变量：
    HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
    LOCAL_MODEL_PATH=~/models/mixtral-8x7b

注意：
    - 不支持 compare 模式（需加载两次模型，显存不够）
    - 每答完一题自动保存 checkpoint 到 ../logs/ 目录
    - 最终汇总也会保存一份到 result/（本地参考用）
"""

import sys, os, re, json, time, argparse, glob

os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import MixtralForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
HOBBIT_CONFIG = {"T1": 0.6, "T2": 0.9}
_mq = os.environ.get("GSM8K_MAX_QUESTIONS", "")
MAX_Q = int(_mq) if _mq else 20
_qs = os.environ.get("GSM8K_QUESTION_START", "")
Q_START = int(_qs) if _qs else 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "gsm8k_test.json")
# checkpoint 放到 ../logs/ 下（服务器只 pull 不 push，../logs/ 不会被 git 覆盖）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# GSM8K 8-shot prompt（标准链式推理样例）
EIGHT_SHOT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then they plant some. 21 trees total. So the number they planted is 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total left?
A: Originally, Leah had 32 and her sister 42. So total was 32 + 42 = 74. After eating 35, they have 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20. Now he has 12. So he gave away 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5. He got 2 from mom and 2 from dad, so 5 + 2 + 2 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were 9 computers. Each day 5 were installed over 4 days: 5 * 4 = 20. Total: 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58. Lost 23 on Tuesday: 58 - 23 = 35. Lost 2 more on Wednesday: 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had $23. Bagels cost $3 * 5 = $15. Left: $23 - $15 = $8. The answer is 8.
"""


def load_model():
    """加载 Mixtral-8x7B（和 server_hobbit.py 一致）"""
    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    local = bool(model_id)
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"

    n_gpus = torch.cuda.device_count()
    print(f"[GSM8K] Loading model ({'local' if local else 'hub'})...")
    model = MixtralForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=(
            {i: "40GB" for i in range(n_gpus)} | ({"cpu": "200GB"} if n_gpus else {})
        ),
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def quantize_weight_to_int4(weight):
    """将权重量化到 INT4 再反量化，返回带量化误差的 float 权重。

    优先用 bitsandbytes（真实 4-bit NF4 量化），
    不可用时用 PyTorch fake quantization 模拟。
    """
    if weight.numel() == 0:
        return weight.clone()

    # 尝试 bitsandbytes 真实量化
    try:
        import bitsandbytes as bnb

        # quantize_4bit: 量化到 4-bit (NF4 格式)，再反量化回 float
        q_data, q_state = bnb.functional.quantize_nf4(weight)
        dq_weight = bnb.functional.dequantize_nf4(q_data, q_state)
        return dq_weight.to(dtype=weight.dtype)
    except Exception:
        pass

    # Fallback: 手动 fake quantization
    min_val = weight.min()
    max_val = weight.max()
    qmin, qmax = 0, 15  # 4-bit: 0~15
    scale = (max_val - min_val) / (qmax - qmin)
    if scale < 1e-12:
        return weight.clone()
    zero_point = -min_val / scale
    q_weight = torch.clamp(torch.round(weight / scale + zero_point), qmin, qmax)
    dq_weight = (q_weight - zero_point) * scale
    return dq_weight


@torch.no_grad()
def make_hobbit_forward(stats, int4_gate_up, int4_down):
    """HOBBIT 决策：INT4 替换 + Skip。支持预量化（GPU 层）和懒量化（meta 层）。"""

    def f(self, hidden_states):
        B, S, D = hidden_states.shape
        x = hidden_states.view(-1, D)
        _, top_k_weights, top_k_index = self.gate(x)
        idx_cpu = top_k_index.cpu().numpy()
        w_cpu = top_k_weights.cpu().numpy()

        int4_experts = set()
        modified_w = top_k_weights.clone()
        device = x.device

        for t in range(B * S):
            tw = w_cpu[t].sum()
            cum = 0.0
            for i in range(len(idx_cpu[t])):
                eid = int(idx_cpu[t][i])
                if i > 0:
                    cum += w_cpu[t][i - 1]
                score = 0.0 if i == 0 else cum / tw
                if score <= HOBBIT_CONFIG["T1"]:
                    stats["hit" if eid in stats["cache"] else "miss"] += 1
                elif score <= HOBBIT_CONFIG["T2"]:
                    stats["int4"] += 1
                    int4_experts.add(eid)
                else:
                    stats["skip"] += 1
                    modified_w[t, i] = 0.0

        # === INT4 替换：设置状态，让 _hobbit_experts_forward 处理 ===
        if int4_experts:
            self.experts._hobbit_int4 = int4_experts
            self.experts._hobbit_int4_weights = (int4_gate_up, int4_down, None)

        x = self.experts(x, top_k_index, modified_w)

        # 清理状态
        if int4_experts:
            self.experts._hobbit_int4 = set()

        return x.reshape(B, S, D)

    return f


# 保存原始 MixtralExperts.forward，后续需要替换
_ORIGINAL_EXPERTS_FORWARD = None


def _patch_experts_forward():
    """全局替换 MixtralExperts.forward，支持 HOBBIT INT4 权重注入。

    HOBBIT 的 mlp.forward 在调用 self.experts() 前设置
    self.experts._hobbit_int4 和 self.experts._hobbit_int4_weights，
    此函数会读取并替换对应专家的权重。
    """
    global _ORIGINAL_EXPERTS_FORWARD
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    if _ORIGINAL_EXPERTS_FORWARD is None:
        _ORIGINAL_EXPERTS_FORWARD = MixtralExperts.forward

    @torch.no_grad()
    def _hobbit_forward(self, hidden_states, top_k_index, top_k_weights):
        int4_set = getattr(self, "_hobbit_int4", None)
        if not int4_set:
            return _ORIGINAL_EXPERTS_FORWARD(self, hidden_states, top_k_index, top_k_weights)

        # 获取 INT4 权重缓存
        q_gateup, q_down, meta_cache = getattr(self, "_hobbit_int4_weights", (None, None, None))
        device = hidden_states.device

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            if expert_idx in int4_set:
                # 使用 INT4 权重
                if meta_cache and expert_idx in meta_cache:
                    g, d = meta_cache[expert_idx]
                else:
                    g, d = q_gateup[expert_idx], q_down[expert_idx]
                gate, up = nn.functional.linear(current_state, g.to(device)).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(
                    current_hidden_states, d.to(device)
                )
            else:
                # 原始 FP16（此时 hooks 已物化权重）
                gate, up = nn.functional.linear(
                    current_state, self.gate_up_proj[expert_idx]
                ).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(
                    current_hidden_states, self.down_proj[expert_idx]
                )

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    MixtralExperts.forward = _hobbit_forward


def _load_meta_int4(model):
    """从 safetensors 读取 meta 层权重，直接预量化为 INT4 并返回缓存。

    不修改模型参数（meta tensor 不可写），只返回供 _hobbit_forward 使用的 INT4 权重。
    """
    meta_layers = [
        (idx, layer) for idx, layer in enumerate(model.model.layers)
        if layer.mlp.experts.gate_up_proj.is_meta
    ]
    if not meta_layers:
        return {}

    model_id = os.environ.get("LOCAL_MODEL_PATH", "").strip()
    if not model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1"
    model_path = os.path.expanduser(model_id) if "~" in model_id else model_id
    if not os.path.isdir(model_path):
        model_path = os.path.join(os.path.expanduser("~"), "models", "mixtral-8x7b")
    if not os.path.isdir(model_path):
        print(f"[LOAD_META] Model path not found: {model_path}")
        return {}

    sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        return {}

    from safetensors import safe_open
    cache = {}  # {expert_id: (int4_gate, int4_down)} on CPU

    print(f"[LOAD_META] Loading & quantizing {len(meta_layers)} meta layers...")
    for idx, layer in meta_layers:
        exp = layer.mlp.experts
        n_exp = exp.gate_up_proj.shape[0]
        hidden_dim = exp.gate_up_proj.shape[-1]
        intermediate_dim = exp.down_proj.shape[-1]

        for sf_path in sf_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for eid in range(n_exp):
                    key = (idx, eid)
                    if key in cache:
                        continue
                    prefix = f"model.layers.{idx}.block_sparse_moe.experts.{eid}"
                    w1k, w2k, w3k = f"{prefix}.w1.weight", f"{prefix}.w2.weight", f"{prefix}.w3.weight"
                    if w1k in keys and w3k in keys:
                        w1, w3 = f.get_tensor(w1k), f.get_tensor(w3k)
                        gate_up = torch.cat([w1, w3], dim=0).to(torch.bfloat16)
                        w2 = f.get_tensor(w2k).to(torch.bfloat16)
                        # 直接预量化到 INT4
                        qg = quantize_weight_to_int4(gate_up)
                        qd = quantize_weight_to_int4(w2)
                        cache[key] = (qg.cpu(), qd.cpu())

        layer_key_count = sum(1 for k in cache if k[0] == idx)
        if layer_key_count > 0:
            print(f"[LOAD_META]   Layer {idx}: {layer_key_count} experts INT4 cached")

    print(f"[LOAD_META] Done — {len(cache)} experts cached")
    return cache


def patch_hobbit(model):
    """给模型缝上 HOBBIT：预计算 INT4 权重 + 替换专家前向。"""
    # 全局替换 MixtralExperts.forward
    _patch_experts_forward()

    # 加载 meta 层的 INT4 权重（不碰模型参数）
    meta_int4 = _load_meta_int4(model)

    n_experts = model.model.layers[0].mlp.experts.gate_up_proj.shape[0]
    dtype = model.model.layers[0].mlp.experts.gate_up_proj.dtype
    hidden_dim = model.model.layers[0].mlp.experts.gate_up_proj.shape[-1]
    intermediate_dim = model.model.layers[0].mlp.experts.down_proj.shape[-1]

    print(f"[HOBBIT] Pre-computing INT4 for {n_experts} experts x {len(model.model.layers)} layers...")
    layer_stats = []

    for layer_idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts

        # GPU 层：预计算 INT4（meta 层走 _load_meta_int4）
        int4_gate_up = torch.empty(n_experts, 2 * intermediate_dim, hidden_dim, device="cpu", dtype=dtype)
        int4_down = torch.empty(n_experts, hidden_dim, intermediate_dim, device="cpu", dtype=dtype)
        for eid in range(n_experts):
            key = (layer_idx, eid)
            if key in meta_int4:
                int4_gate_up[eid], int4_down[eid] = meta_int4[key]
            else:
                int4_gate_up[eid] = quantize_weight_to_int4(experts.gate_up_proj.data[eid]).cpu()
                int4_down[eid] = quantize_weight_to_int4(experts.down_proj.data[eid]).cpu()

        if (layer_idx + 1) % 8 == 0:
            print(f"[HOBBIT]   Layer {layer_idx+1}/{len(model.model.layers)} processed")

        s = {"hit": 0, "miss": 0, "int4": 0, "skip": 0, "cache": {0, 1}}
        layer_stats.append(s)
        layer.mlp.forward = make_hobbit_forward(s, int4_gate_up, int4_down).__get__(
            layer.mlp, type(layer.mlp)
        )

    print(f"[HOBBIT] Patch complete — {len(model.model.layers)} layers")
    # 返回可聚合的 stats 对象

    class HobbitStats:
        def __init__(self, stats_list):
            self._list = stats_list

        def values(self):
            total = self.aggregate()
            return {k: v for k, v in total.items() if k != "cache"}

        def aggregate(self):
            total = {"hit": 0, "miss": 0, "int4": 0, "skip": 0}
            for s in self._list:
                for k in ("hit", "miss", "int4", "skip"):
                    total[k] += s[k]
            return total

        def __getitem__(self, key):
            return self.aggregate()[key]

        def __contains__(self, key):
            return key in self.aggregate()

    return HobbitStats(layer_stats)


def extract_answer(text):
    """从生成的文本中提取最终答案（#### 后的数字）"""
    # 优先找 "####" 后的数字
    m = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if m:
        ans = m.group(1).strip()
        # 去掉末尾小数点，如 "18." -> "18"
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    # 备选：找 "answer is" 后的数字
    m = re.search(r"[Aa]nswer\s+is\s*(-?\d+\.?\d*)", text)
    if m:
        ans = m.group(1).strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    # 备选：找最后出现的数字
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        ans = nums[-1].strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    return ""


def evaluate_gsm8k(
    model,
    tokenizer,
    questions,
    answers,
    device,
    max_new_tokens=256,
    checkpoint_path=None,
    resume=False,
):
    """在 GSM8K 问题上跑推理，支持增量 checkpoint + 断点续跑。

    每答完一题立即保存 checkpoint，进程崩溃后可 --resume 续跑。
    """
    results = []
    start_idx = 0

    # 续跑：加载已有 checkpoint，跳过已完成的题目
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        results = cp.get("results", [])
        start_idx = len(results)
        if start_idx > 0:
            print(
                f"[GSM8K]   Resuming from checkpoint: {start_idx}/{len(questions)} done"
            )

    t0 = time.time()

    for i in range(start_idx, len(questions)):
        q, gt = questions[i], answers[i]
        prompt = EIGHT_SHOT_EXAMPLES + f"Q: {q}\nA:"
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

        # 提取生成的回答（去掉 prompt 部分）
        input_len = inp["input_ids"].shape[1]
        generated = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        pred = extract_answer(generated)
        correct = pred == gt

        results.append(
            {
                "idx": i + Q_START,
                "question": q,
                "ground_truth": gt,
                "prediction": pred,
                "generated": generated.strip(),
                "correct": correct,
            }
        )

        # 每答完一题立即保存 checkpoint
        if checkpoint_path:
            cp = {
                "mode": "in_progress",
                "total": len(questions),
                "completed": i + 1,
                "start_idx": Q_START,
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(cp, f, ensure_ascii=False, indent=2)

        dt = time.time() - t0
        correct_so_far = sum(r["correct"] for r in results)
        elapsed_per_q = dt / (i + 1 - start_idx) if (i + 1 - start_idx) > 0 else 0
        eta = elapsed_per_q * (len(questions) - i - 1)
        print(
            f"[GSM8K]   #{i+1}/{len(questions)} | "
            f"acc={correct_so_far/(i+1)*100:.1f}% | "
            f"{dt/(i+1):.1f}s/q | "
            f"ETA={eta/60:.0f}min"
        )

    return results


def print_results(results, label, elapsed):
    """打印评测结果汇总"""
    correct = sum(r["correct"] for r in results)
    total = len(results)
    acc = correct / total * 100

    print(f"\n{'='*60}")
    print(f"[{label}] Results: {correct}/{total} = {acc:.1f}%")
    print(f"[{label}] Time: {elapsed:.1f}s ({elapsed/total:.1f}s/q)")
    print(f"{'='*60}")

    # 打印前 5 个错误示例
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n[{label}] Sample errors ({min(5, len(errors))} shown):")
        for r in errors[:5]:
            idx = r["idx"]
            q_short = r["question"][:60]
            print(f"  #{idx}: GT={r['ground_truth']}, Pred={r['prediction']}")
            print(f"       Q: {q_short}...")

    return acc


def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hobbit"],
        required=True,
        help="baseline (全FP16) 或 hobbit (混合精度)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="从 checkpoint 续跑（跳过已完成的题目）"
    )
    args = parser.parse_args()
    mode = args.mode

    # 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"[GSM8K] Data file not found: {DATA_FILE}")
        print("[GSM8K] Run: python tools/download_gsm8k.py")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    total_available = len(all_data)
    n_questions = min(MAX_Q, total_available - Q_START)
    subset = all_data[Q_START : Q_START + n_questions]
    questions = [item["question"] for item in subset]
    answers = [item["answer"] for item in subset]

    # checkpoint 文件放 ../logs/ 下（服务器 git pull 安全区）
    cp_name = f"gsm8k_checkpoint_{mode}_{Q_START}-{Q_START+n_questions-1}.json"
    checkpoint_path = os.path.join(LOG_DIR, cp_name)

    print(f"{'='*60}")
    print(
        f"GSM8K Evaluation — Mode: {mode.upper()}"
        + (" (resume)" if args.resume else "")
    )
    print(
        f"Questions: {Q_START}-{Q_START + n_questions - 1} ({n_questions}/{total_available})"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # ---- 加载模型 ----
    model, tokenizer = load_model()
    device = model.device

    # HOBBIT 模式：缝补丁
    hobbit_stats = None
    if mode == "hobbit":
        hobbit_stats = patch_hobbit(model)

    # ---- 跑推理 ----
    t0 = time.time()
    results = evaluate_gsm8k(
        model,
        tokenizer,
        questions,
        answers,
        device,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
    elapsed = time.time() - t0
    acc = print_results(results, mode.upper(), elapsed)

    # ---- 保存最终结果 ----
    # 1) 保存到 ../logs/（服务器使用）
    final_out_log = os.path.join(
        LOG_DIR, f"gsm8k_{mode}_{Q_START}-{Q_START+n_questions-1}.json"
    )
    save_data = {
        "mode": mode,
        "accuracy": acc,
        "n_questions": n_questions,
        "start_idx": Q_START,
        "elapsed": elapsed,
        "results": results,
    }
    if hobbit_stats:
        agg = hobbit_stats.aggregate()
        save_data["hobbit_stats"] = agg
        stat_keys = [k for k in agg if k != "cache"]
        total_calls = sum(agg[k] for k in stat_keys)
        print(f"\n[HOBBIT Stats]")
        print(f"  FP16 hit:  {agg['hit']:>6} ({agg['hit']/total_calls*100:5.1f}%)")
        print(f"  FP16 miss: {agg['miss']:>6} ({agg['miss']/total_calls*100:5.1f}%)")
        print(f"  INT4:      {agg['int4']:>6} ({agg['int4']/total_calls*100:5.1f}%)")
        print(f"  Skip:      {agg['skip']:>6} ({agg['skip']/total_calls*100:5.1f}%)")
        print(f"  INT4+Skip: {(agg['int4']+agg['skip'])/total_calls*100:.1f}% of calls")

    with open(final_out_log, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"[GSM8K] Saved to {final_out_log}")

    # 2) 也保存一份到 result/（本地 pull 下来后参考）
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_out_res = os.path.join(
        RESULT_DIR, f"gsm8k_{mode}_{Q_START}-{Q_START+n_questions-1}.json"
    )
    with open(final_out_res, "w") as f:
        json.dump(save_data, f, indent=2)

    # 清理 checkpoint（已完成，不再需要续跑）
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[GSM8K] Checkpoint cleaned: {cp_name}")

    print(f"\n[GSM8K] DONE")


if __name__ == "__main__":
    main()
