import sys,os
import torch
from torchvision import transforms,models
from torch.utils.mobile_optimizer import optimize_for_mobile

from PIL import Image
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import timm
print(pretrainedmodels.model_names)

def get_model(model_name, model_class):
    if 'efficientnet' in model_name:
        #return timm.create_model(model_name, pretrained=True)
        #model=torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        model=EfficientNet.from_pretrained(model_name)
        model.set_swish(False)
        return model
        #return EfficientNet.from_pretrained(model_name)
    elif model_name.startswith('pretrained_'):
        print(model_class)
        return model_class(num_classes=1000, pretrained='imagenet')
    else:
        return model_class(pretrained=True)
if __name__ == '__main__':
    output_model_dir='app/src/main/assets'
    INPUT_SIZE=224
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    all_models={'mobilenet_v2':models.mobilenet_v2,'resnet18':models.resnet18,
        #'mnasnet':models.mnasnet1_0, 'pretrained_nasnetmobile':pretrainedmodels.nasnetamobile,
        'pretrained_squeezenet':pretrainedmodels.squeezenet1_1,
        #'tf_efficientnet_b0_ns':None
        'efficientnet-b0':None
        }
    
    for model_name in all_models:
        filename=os.path.join(output_model_dir,model_name)
        if not os.path.exists(filename+'.ptl'):
            model = get_model(model_name,all_models[model_name])
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
            traced_script_module = torch.jit.trace(model, example)
            #traced_script_module.save(filename+'.pt')
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(filename+'.ptl')

            traced_script_module = torch.jit.trace(quantized_model, example)
            #traced_script_module.save(filename+'_quant.pt')
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(filename+'_quant.ptl')
