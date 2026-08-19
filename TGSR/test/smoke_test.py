import torch

from model import TGSR


def main() -> None:
    model = TGSR(upscale=4).eval()
    lr = torch.rand(1, 3, 32, 32)
    with torch.inference_mode():
        sr, aux = model(lr, return_vis=True)

    assert sr.shape == (1, 3, 128, 128)
    assert aux["tde"]["assignments"].shape[:2] == (1, 32 * 32)
    assert aux["tde"]["states"].shape[1] == 16
    print("TGSR smoke test passed.")


if __name__ == "__main__":
    main()
