from utils.train import train

if __name__ == "__main__":
    train(num_classes=91, epochs=10, lr=1e-4, batch_size=8, device="cuda")
