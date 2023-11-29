import argparse
from model import GAN

def main():
    parser = argparse.ArgumentParser(description='Train GAN model.')
    parser.add_argument('train_dir', type=str, help='Training directory path')
    parser.add_argument('current_epoch', type=int, help='Current epoch number')
    parser.add_argument('num_epochs', type=int, help='Total number of epochs')
    parser.add_argument('resize', type=int, help='Size of the image in pixels')

    args = parser.parse_args()

    # Use the provided arguments
    gan = GAN(f"data/train_{args.train_dir}", "data/val_1", current_epoch=args.current_epoch, num_epochs=args.num_epochs, resize=args.resize)
    gan.train()

if __name__ == '__main__':
    main()
