"""Phase 1 sanity: v1 weights in the v3 model (max_len 128, zero-init quote/glyph embeddings, 64-token null seq)
must reproduce the original v1/v2 code (max_len 64) exactly on prompts that fit in 64 chars.
Run: python tests/test_warmstart.py [--teacher weights/v1_teacher.safetensors] [--v2_dir ../grinpress-v2]"""
import argparse, importlib.util, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch


def load_v2_modules(v2_dir):
    """Import grinpress-v2's conditioning.py + model.py under private names (they import 'conditioning')."""
    saved = sys.modules.get("conditioning")
    spec = importlib.util.spec_from_file_location("v2_conditioning", os.path.join(v2_dir, "conditioning.py"))
    c2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c2)
    sys.modules["conditioning"] = c2
    try:
        spec = importlib.util.spec_from_file_location("v2_model", os.path.join(v2_dir, "model.py"))
        m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
    finally:
        if saved is not None: sys.modules["conditioning"] = saved
        else: sys.modules.pop("conditioning")
    return c2, m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(ROOT, "weights", "v1_teacher.safetensors"))
    ap.add_argument("--v2_dir", default=os.path.join(os.path.dirname(ROOT), "grinpress-v2"))
    ap.add_argument("--euler_steps", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(os.cpu_count())
    c2, m2 = load_v2_modules(a.v2_dir)
    import conditioning as c3, model as m3
    from eval_grid import grid48, grid14
    v1 = m2.FlowModel().eval(); m2.load_teacher_weights(v1, a.teacher)
    v3 = m3.FlowModel().eval(); kept = m3.load_teacher_weights(v3, a.teacher)
    caps = [c for _, c, _ in grid48() + grid14() if len(c) <= 64][:12]
    B = len(caps)
    x = torch.randn(B, 3, 128, 128); t = torch.rand(B)
    i1, k1 = c2.tokenize(caps); i3, k3 = c3.tokenize(caps)
    assert torch.equal(i1, i3[:, :64]) and int(k3[:, 64:].sum()) == 0
    res = {"n_prompts": B, "params_v1": m3.count_params(v1), "params_v3": m3.count_params(v3), "left_at_init": kept}
    with torch.no_grad():
        ctx1, mm1 = v1.cond(i1, k1); ctx3, mm3 = v3.cond(i3, k3)
        res["ctx_maxdiff_real_tokens"] = float((ctx1 - ctx3[:, :64]).masked_fill(~k1[..., None], 0).abs().max())
        vc1 = v1(x, t, i1, k1); vc3 = v3(x, t, i3, k3)
        vu1 = v1(x, t, i1, k1, force_uncond=True); vu3 = v3(x, t, i3, k3, force_uncond=True)
        res["v_cond_maxdiff"] = float((vc1 - vc3).abs().max())
        res["v_uncond_maxdiff"] = float((vu1 - vu3).abs().max())
        res["v_scale_meanabs"] = float(vc1.abs().mean())
        # short CFG-5 Euler sample from identical noise
        z = torch.randn(4, 3, 128, 128, generator=torch.Generator().manual_seed(7))
        xs = []
        for mdl, ids, mk in ((v1, i1[:4], k1[:4]), (v3, i3[:4], k3[:4])):
            xx = z.clone(); ts = torch.linspace(1, 0, a.euler_steps + 1)
            for j in range(a.euler_steps):
                tb = ts[j].expand(4)
                v = mdl(xx, tb, ids, mk, force_uncond=True) + 5.0 * (mdl(xx, tb, ids, mk) - mdl(xx, tb, ids, mk, force_uncond=True))
                xx = xx - (ts[j] - ts[j + 1]) * v
            xs.append(xx.clamp(-1, 1))
        res[f"cfg5_{a.euler_steps}step_image_maxdiff_pixels"] = float((xs[0] - xs[1]).abs().max() * 127.5)
    tol = 1e-4
    res["PASS"] = res["v_cond_maxdiff"] < tol and res["v_uncond_maxdiff"] < tol and res["ctx_maxdiff_real_tokens"] < tol \
        and res[f"cfg5_{a.euler_steps}step_image_maxdiff_pixels"] < 0.5
    json.dump(res, open(os.path.join(ROOT, "assets", "phase1_warmstart_test.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert res["PASS"]
    print("PASS")


if __name__ == "__main__":
    main()
