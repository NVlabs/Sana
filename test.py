import torch
from app.sana_sprint_pipeline import SanaSprintPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaSprintPipeline("configs/sana_sprint_config/1024ms/SanaSprint_600M_1024px_allqknorm_bf16_scm_ladd_dc_ae_turbo.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/Sana_Sprint_0.6B_1024px/checkpoints/Sana_Sprint_0.6B_1024px.pth")
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

image = sana(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=generator,
)
save_image(image, 'output/sana.png', nrow=1, normalize=True, value_range=(-1, 1))