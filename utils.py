import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import torch
from utils.inception_score import evaluate_inception_score
import glob
import pandas as pd    



def sample_image(model, encoder, output_image_dir, n_row, batches_done, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(conditional_embeddings).cpu()
            gen_imgs.append(imgs)

            if len(captions) > n_row ** 2:
                done = True
                break

    gen_imgs = torch.cat(gen_imgs).numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
    plt.close()


def load_model(file_path, generative_model):
    dict = torch.load(file_path)
    generative_model.load_state_dict(dict['G'])
    
    
    
    
def calc_IS(model, encoder, n_row, dataloader, device):
    """Saves a grid of generated imagenet pictures with captions"""
#    target_dir = os.path.join(output_image_dir, "samples/")
#    if not os.path.isdir(target_dir):
#        os.makedirs(target_dir)

    captions = []
    gen_imgs = []
    # get sample captions
    done = False
    while not done:
        for (_, labels_batch, captions_batch) in dataloader:
            captions += captions_batch
            conditional_embeddings = encoder(labels_batch.to(device), captions)
            imgs = model.sample(conditional_embeddings).cpu()
            gen_imgs.append(imgs)

            if len(captions) > n_row:
                done = True
                break
    imgs = imgs * 2 - 1
    iscore, _ = evaluate_inception_score(imgs)
    return iscore




def evaluate(model_G, embedder, test_loader, device, p):
#    assert opt.model_checkpoint is not None, 'no model checkpoint specified'
    model_paths = sorted(glob.glob(p + "/*.pt"))

    epochs = []
    inception_scores = []

    print(f"Calculating All Inception Scores...  (# {len(model_paths)})")
    for i, path in enumerate(model_paths):
        load_model(path, model_G)
        iscore = calc_IS(model_G, embedder, 10000, test_loader, device)
        epoch = int(os.path.basename(path).replace("epoch_", "").replace(".pt", ""))
        epochs.append(epoch)
        inception_scores.append(iscore)
        print(f"epoch = {epoch}, inception_score = {iscore}    [{i+1}/{len(model_paths)}]")

    df = pd.DataFrame({"epoch": epochs, "inception_score": inception_scores})
    df.to_csv(p +"/inception_score.csv", index=False)




