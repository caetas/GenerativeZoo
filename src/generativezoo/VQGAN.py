from models.AR.VQGAN import VQModel

model = VQModel(
    in_channels=3,
    out_channels=3,
    channels=64,
    z_channels=64,
    resolution=256,
    ch_mult=[1, 2, 2, 2, 2],
    num_res_blocks=2,
    attn_resolutions=[2, 4],
    dropout=0.1,
    double_z=False,
    n_embed=1024,
    embed_dim=64)

model.train_model(None, None)

