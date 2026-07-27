"""仿真系统 CLI 入口"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from simulator.scenario import ScenarioEngine
from simulator.injectors import RedisInjector, inject_ws_async

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def cmd_list(_args):
    """列出可用场景"""
    if not SCENARIOS_DIR.exists():
        print("无场景目录")
        return
    yamls = sorted(SCENARIOS_DIR.glob("*.yaml"))
    if not yamls:
        print("无场景文件")
        return
    for f in yamls:
        print(f"  {f.stem}  ({f.name})")


def cmd_run(args):
    """运行场景"""
    # 1. 加载并编排
    print(f"加载场景: {args.scenario}")
    engine = ScenarioEngine(
        args.scenario,
        speed=args.speed,
        params_path=args.params,
        dataset_path=args.dataset,
    )
    frames = engine.run()
    print(f"生成 {len(frames)} 帧 (speed={args.speed}x)")

    # 2. Ground truth 落盘
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_path = out_dir / "ground_truth.jsonl"
        count = 0
        with open(gt_path, "w") as f:
            for fg in frames:
                if fg.gt:
                    f.write(json.dumps(fg.gt.to_dict(), ensure_ascii=False) + "\n")
                    count += 1
        print(f"Ground truth → {gt_path} ({count} 条, {gt_path.stat().st_size} bytes)")

    # 3. 双通道注入
    if args.redis:
        print(f"Redis 注入: {args.redis}")
        try:
            ri = RedisInjector(args.redis)
            ri.inject(frames)
            ri.close()
            print("  ✓ 完成")
        except Exception as e:
            print(f"  ✗ Redis 注入失败: {e}")

    if args.ws:
        print(f"WebSocket 注入: {args.ws}")
        device_id = args.device or "sim_device"
        secret = args.secret or "sim_secret"
        try:
            asyncio.run(inject_ws_async(frames, args.ws, device_id, secret))
            print("  ✓ 完成")
        except Exception as e:
            print(f"  ✗ WS 注入失败: {e}")

    if not args.redis and not args.ws and not args.out:
        print("(未指定输出目标: --out, --redis, --ws)")


def cmd_calibrate(args):
    """校准参数"""
    from simulator.calibrate import calibrate, save_params
    params = calibrate(args.dataset, k=args.k)
    if not params:
        print("[ERROR] 没有拟合出任何动作参数")
        sys.exit(1)
    save_params(params, args.out)
    print(f"参数已保存到: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="housafe 仿真系统")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="列出可用场景")

    # run
    p_run = sub.add_parser("run", help="运行场景")
    p_run.add_argument("scenario", help="场景 YAML 路径")
    p_run.add_argument("--speed", type=float, default=1.0,
                       help="时间加速倍率 (default: 1, 10 表示 10x 加速)")
    p_run.add_argument("--params", help="校准参数 npz 路径")
    p_run.add_argument("--dataset", help="3DPCHM 数据集 npz 路径（replay 模式用）")
    p_run.add_argument("--out", help="Ground truth 输出目录")
    p_run.add_argument("--redis", help="Redis URL (如 redis://localhost:6379/0)")
    p_run.add_argument("--ws", help="WebSocket URL (如 ws://localhost:8000/ws/ingest)")
    p_run.add_argument("--device", help="设备 ID")
    p_run.add_argument("--secret", help="设备密钥")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="校准 GMM 参数")
    p_cal.add_argument("--dataset", required=True, help="3DPCHM 预处理 npz 路径")
    p_cal.add_argument("--out", required=True, help="输出参数文件路径")
    p_cal.add_argument("--k", type=int, default=4, help="GMM 分量数 (default: 4)")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
