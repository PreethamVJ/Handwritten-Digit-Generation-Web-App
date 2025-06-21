import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- Model Definition (same as training) ---
class CVAE(torch.nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, num_classes=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.fc1 = torch.nn.Linear(input_dim + num_classes, 400)
        self.fc21 = torch.nn.Linear(400, latent_dim)
        self.fc22 = torch.nn.Linear(400, latent_dim)

        self.fc3 = torch.nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = torch.nn.Linear(400, input_dim)

    def encode(self, x, y):
        one_hot = F.one_hot(y, self.num_classes).float()
        x = torch.cat([x, one_hot], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        one_hot = F.one_hot(y, self.num_classes).float()
        z = torch.cat([z, one_hot], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Generate Images Function ---
def generate_images(model_path, digit, num_samples=5, device='cpu'):
    model = CVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y = torch.full((num_samples,), digit, dtype=torch.long, device=device)
    z = torch.randn(num_samples, model.latent_dim, device=device)

    with torch.no_grad():
        images = model.decode(z, y).cpu().view(-1, 28, 28)
    return images

# --- Streamlit UI ---
st.set_page_config(page_title="Handwritten Digit Image Generator")
st.title("🖋️ Handwritten Digit Image Generator")

st.markdown("Generate synthetic MNIST-like digit images using your trained CVAE model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))
generate_btn = st.button("Generate Images")

if generate_btn:
    st.subheader(f"Generated 5 samples of digit **{digit}**")
    images = generate_images("cvae_mnist.pth", digit=int(digit))

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Sample {i+1}")
    st.pyplot(fig)
