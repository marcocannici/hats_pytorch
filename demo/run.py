import time
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from hats_pytorch import HATS
from demo.dataset import PseeDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    hats = HATS((100, 120), r=3, k=10, tau=1e9, delta_t=100000, fold=True)
    hats.to('cuda:0')
    hats.eval()

    dataset = PseeDataset(args.data_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=collate_fn)

    t_tot = 0
    with torch.no_grad():
        for lens, events in tqdm(loader):
            events = events.to('cuda:0')
            lens = lens.to('cuda:0')
            t_rec = time.time()
            hists = hats(events, lens)
            t_tot += time.time() - t_rec

            if args.show:
                for hist in hists:
                    plt.figure(figsize=(10, 8))
                    img = hist[0].cpu().numpy()
                    plt.imshow(img, cmap='hot')
                    plt.show()

    print("Mean time: {} sec".format(t_tot / len(dataset)))


if __name__ == "__main__":
    main()
