import nemo
import nemo.collections.asr as nemo_asr

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
