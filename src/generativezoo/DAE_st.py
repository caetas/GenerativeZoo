from models.DDPM.MONAI_DiffAE import DiffAE
import torch
import os
import streamlit as st
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from PIL import Image
import numpy as np
import base64
import io
from PIL import Image

def pneumoniamnist_train_loader(batch_size, normalize=True, size = 64):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    dataset = PneumoniaMNIST(root='./../../data/raw', split='train', download=True, transform=transform, size = size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return loader

def pneumoniamnist_val_loader(batch_size, normalize=True, size = 64):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    dataset = PneumoniaMNIST(root='./../../data/raw', split='val', download=True, transform=transform, size = size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return loader

@st.cache_resource
def get_train_loader():
    return pneumoniamnist_train_loader(16, normalize=True, size = 64)

@st.cache_resource
def get_val_loader():
    return pneumoniamnist_val_loader(16, normalize=True, size = 64)

# create a streamlit function to get the model
@st.cache_resource
def get_model():
    model = DiffAE(embedding_dimension=512, num_train_timesteps=1000, inference_timesteps=200, lr=2e-5, num_epochs = 100, in_channels=1)
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    model.load_state_dict(torch.load(os.path.join('./../../models', 'DiffAE_best_model.pth')))
    model.linear_regression(train_loader, val_loader)
    return model

# create a function that loads a different sample from the validation set
@st.cache_resource
def validation_samples(_val_loader):
    batch = next(iter(_val_loader))
    return batch[0][:,:,:], batch[1][:]

def manipulate_latent(_model, _transformation, _image):
    # add a message to the streamlit app saying that it is being generated
    st.write("Generating the transformed image...")
    image, label = _model.manipulate_image(_image, _transformation)
    return image, label

# start streamlit
st.title("Diffusion Autoencoder")
st.write("This is a simple example of a Diffusion Autoencoder. The model is trained on the PneumoniaMNIST dataset")

model = get_model()

#sample, label = validation_sample(get_val_loader())

samples,labels = validation_samples(get_val_loader())

# select one of the samples
sample_number = st.selectbox("Select a sample", range(len(samples)))

sample = samples[sample_number]
label = labels[sample_number]

if sample is not None:
    image = sample.cpu().detach().numpy()*0.5 + 0.5
    image = image.clip(0, 1)
    image = np.transpose(image, (1,2,0))
    image = (image*255).astype(np.uint8)
    image = Image.fromarray(image.squeeze())
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    if label == 0:
        caption = 'Normal'
    else:
        caption = 'Pneumonia'

    # show the sample
    st.write(f"Label: {caption}")
    # show the image with a width of 300 pixels and centered
    #st.image(image, width=300)
    # Center the image using CSS styling
    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/jpg;base64,{base64.b64encode(img_bytes).decode()}" width="{300}">'
        f'</div>',
        unsafe_allow_html=True
    )

    # create a slider to transform the image, should go from -2 to 2. Should say Normal next to -2 and Pneumonia next to 2
    st.write("Transform the image")
    transformation = st.slider('Transformation', -0.5, 0.5, 0.0, 0.05, help="This slider will transform the image. -0.5 should make it more Normal and 0.5 should make it look more like Pneumonia")

    # get a button to transform the image
    if st.button("Transform"):
        transformed_image, new_label = manipulate_latent(model, transformation, sample)
        transformed_image = (transformed_image*255).astype(np.uint8)
        transformed_image = Image.fromarray(transformed_image.squeeze())
        transformed_img_bytes = io.BytesIO()
        transformed_image.save(transformed_img_bytes, format='JPEG')
        transformed_img_bytes = transformed_img_bytes.getvalue()
        #st.image(transformed_image, width=300)

        # Center the image using CSS styling
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="data:image/jpg;base64,{base64.b64encode(transformed_img_bytes).decode()}" width="{300}">'
            f'</div>',
            unsafe_allow_html=True
        )

        if new_label == 0:
            new_caption = 'Normal'
        else:
            new_caption = 'Pneumonia'

        st.write(f"Label: {new_caption}")