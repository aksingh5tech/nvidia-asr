from nemo.collections.asr.models import EncDecCTCModelBPE

model = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
model.save_to("parakeet_tdt_0.6b_v2.nemo")

if __name__ == '__main__':
    pass