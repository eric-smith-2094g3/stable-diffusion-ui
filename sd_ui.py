import argparse
import json
import os
import sys
from pathlib import Path

from sd_client import SDClient, SDError

def _load_config():
    cfg = {}
    cfg_path = Path.home() / ".config" / "sd_ui" / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return cfg

def _save_config(cfg):
    cfg_dir = Path.home() / ".config" / "sd_ui"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

def _default_url():
    return os.environ.get("SD_URL", "http://127.0.0.1:7860")

def cmd_generate(args):
    client = SDClient(_default_url())
    cfg = _load_config()

    prompt = args.prompt
    if args.file:
        prompt = Path(args.file).read_text().strip()

    if not prompt:
        print("error: prompt is empty", file=sys.stderr)
        sys.exit(1)

    params = {
        "prompt": prompt,
        "negative_prompt": args.negative or cfg.get("negative_prompt", ""),
        "steps": args.steps or cfg.get("steps", 20),
        "width": args.width or cfg.get("width", 512),
        "height": args.height or cfg.get("height", 512),
        "cfg_scale": args.cfg or cfg.get("cfg_scale", 7.0),
        "sampler_name": args.sampler or cfg.get("sampler_name", "Euler a"),
        "seed": args.seed if args.seed is not None else cfg.get("seed", -1),
    }

    out_dir = Path(args.output) if args.output else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = client.txt2img(params, out_dir=out_dir, save_preview=args.preview)
    except SDError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"saved: {result}")

def cmd_queue(args):
    client = SDClient(_default_url())
    cfg = _load_config()

    prompts = []
    if args.file:
        lines = Path(args.file).read_text().splitlines()
        prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    elif args.prompts:
        prompts = args.prompts
    else:
        print("error: provide --file or positional prompts", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "negative_prompt": args.negative or cfg.get("negative_prompt", ""),
        "steps": args.steps or cfg.get("steps", 20),
        "width": args.width or cfg.get("width", 512),
        "height": args.height or cfg.get("height", 512),
        "cfg_scale": args.cfg or cfg.get("cfg_scale", 7.0),
        "sampler_name": args.sampler or cfg.get("sampler_name", "Euler a"),
        "seed": -1,
    }

    failed = 0
    for i, prompt in enumerate(prompts):
        params = {**base_params, "prompt": prompt}
        print(f"[{i+1}/{len(prompts)}] generating: {prompt[:60]}...")
        try:
            result = client.txt2img(params, out_dir=out_dir, save_preview=args.preview)
            print(f"  -> {result}")
        except SDError as e:
            print(f"  -> failed: {e}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed}/{len(prompts)} failed", file=sys.stderr)
        sys.exit(1)

def cmd_status(args):
    client = SDClient(_default_url())
    try:
        state = client.progress()
    except SDError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if state["progress"] == 0 and not state.get("active"):
        print("idle")
        return

    print(f"progress: {state['progress']*100:.1f}%")
    if state.get("eta"):
        print(f"eta: {state['eta']:.1f}s")
    if state.get("current_image"):
        print("preview: available (use --preview to save)")

def cmd_config(args):
    cfg = _load_config()
    if args.set:
        for pair in args.set:
            if "=" not in pair:
                print(f"error: bad key=value: {pair}", file=sys.stderr)
                sys.exit(1)
            k, v = pair.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            cfg[k] = v
        _save_config(cfg)
        print("saved")
    elif args.get:
        print(cfg.get(args.get, "(not set)"))
    else:
        for k, v in cfg.items():
            print(f"{k}={json.dumps(v)}")

def cmd_samplers(args):
    client = SDClient(_default_url())
    try:
        for name in client.samplers():
            print(name)
    except SDError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(prog="sd_ui")
    sub = parser.add_subparsers(dest="command")

    p_gen = sub.add_parser("generate", help="single image generation")
    p_gen.add_argument("prompt", nargs="?", help="prompt text")
    p_gen.add_argument("--file", "-f", help="read prompt from file")
    p_gen.add_argument("--negative", "-n", help="negative prompt")
    p_gen.add_argument("--steps", type=int, help="sampling steps")
    p_gen.add_argument("--width", type=int, help="image width")
    p_gen.add_argument("--height", type=int, help="image height")
    p_gen.add_argument("--cfg", type=float, help="CFG scale")
    p_gen.add_argument("--sampler", help="sampler name")
    p_gen.add_argument("--seed", type=int, help="seed (-1 for random)")
    p_gen.add_argument("--output", "-o", help="output directory")
    p_gen.add_argument("--preview", action="store_true", help="save preview image")

    p_queue = sub.add_parser("queue", help="batch generation from prompts")
    p_queue.add_argument("prompts", nargs="*", help="prompt strings")
    p_queue.add_argument("--file", "-f", help="file with one prompt per line")
    p_queue.add_argument("--negative", "-n", help="negative prompt")
    p_queue.add_argument("--steps", type=int, help="sampling steps")
    p_queue.add_argument("--width", type=int, help="image width")
    p_queue.add_argument("--height", type=int, help="image height")
    p_queue.add_argument("--cfg", type=float, help="CFG scale")
    p_queue.add_argument("--sampler", help="sampler name")
    p_queue.add_argument("--output", "-o", help="output directory")
    p_queue.add_argument("--preview", action="store_true", help="save preview images")

    sub.add_parser("status", help="show current generation status")
    sub.add_parser("samplers", help="list available samplers")

    p_cfg = sub.add_parser("config", help="get/set defaults")
    p_cfg.add_argument("--set", action="append", metavar="KEY=VALUE", help="set config value")
    p_cfg.add_argument("--get", metavar="KEY", help="get config value")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "queue":
        cmd_queue(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "samplers":
        cmd_samplers(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_usage()
        sys.exit(2)

if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        sys.exit(130)
