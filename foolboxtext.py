import torchvision.models as models
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
 
    model = models.resnet101(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
    clean_acc=fb.utils.accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")


    attack = fb.attacks.LinfDeepFoolAttack()
    epsilons = np.linspace(0.0, 0.005, num=20)
    
    images = ep.astensor(images)
    labels = ep.astensor(labels)
    criterion = fb.criteria.Misclassification(labels)

    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
 
    robust_accuracy = 1 - is_adv.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    plt.plot(epsilons, robust_accuracy.numpy())
    

if __name__ == "__main__":
    main()


    