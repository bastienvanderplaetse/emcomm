import torch
import torch.optim as optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from dataset import RandomRelationalDataset
from speaker import Speaker
from listener import Listener
from losses import ReinforceSpeakerLoss, CPCListenerLoss, ImitationCrossEntropyLoss
from sampling import sample
from EMA import EMA, EMA_score
import random 
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from extractor import Extractor

torch.cuda.init()

resnet = 50 # 50 for ResNet-50 / 18 for ResNet-18
summary_name = "models-name" # Summary name for Tensorboard
models_dir = "models-name" # Directory to store models
f_i = "avg" # avg for average pooling / flatten for flattening / custom for flattening with custom initialization

save_step = 200 # Number of steps before saving the current version of the models


batch_size = 20
kl_EMA = 0.99
imit_EMA = 0.99
imitation_step = 1
n_students = 4
population_size = 10
lr = 0.0001
betas = (0.9, 0.999)
eps = 1e-08
max_steps = int(5e5)
speaker_entropy = 1e-4
speaker_kl_target = 0.5
msg_length = 10
vocab_size = 20
emb_dim = 10

if resnet == 50:
    if f_i == 'avg':
        emb_img = 2048
    else:
        emb_img = 100352
elif resnet == 18:
    if f_i == 'avg':
        emb_img = 512
    else:
        emb_img = 25088

core_state_dim_speaker = 512
core_state_dim_listener = 1024
target_proj_dim = 256
core_state_proj_dim = 256
valid_interval = 200
seed = 1234

w = tensorboard.SummaryWriter(log_dir=os.path.join("tensorboard", summary_name))


def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

fix_seed(seed)

train_combos = [('circlefill', 'cross', 'topleft'), ('circlefill', 'square', 'topright'), ('squarefill', 'squarefill', 'topleft'), ('cross', 'squarefill', 'right'), ('squarefill', 'square', 'top'), ('circle', 'circlefill', 'right'), ('squarefill', 'cross', 'top'), ('square', 'circlefill', 'topright'), ('squarefill', 'squarefill', 'right'), ('squarefill', 'square', 'topright'), ('squarefill', 'cross', 'right'), ('squarefill', 'squarefill', 'topright'), ('square', 'squarefill', 'right'), ('circlefill', 'circle', 'topleft'), ('square', 'cross', 'right'), ('circle', 'square', 'top'), ('squarefill', 'cross', 'topright'), ('square', 'cross', 'topright'), ('cross', 'circle', 'top'), ('squarefill', 'circle', 'right'), ('circlefill', 'cross', 'topright'), ('circle', 'circlefill', 'topright'), ('squarefill', 'circle', 'topleft'), ('squarefill', 'circle', 'topright'), ('squarefill', 'circlefill', 'topleft'), ('circlefill', 'squarefill', 'top'), ('circlefill', 'cross', 'right'), ('circlefill', 'circle', 'topright'), ('circle', 'square', 'right'), ('circle', 'circle', 'topleft'), ('circlefill', 'circlefill', 'topright'), ('cross', 'circle', 'topright'), ('cross', 'circlefill', 'top'), ('squarefill', 'circle', 'top'), ('circlefill', 'squarefill', 'topright'), ('circlefill', 'circle', 'top'), ('circle', 'square', 'topleft'), ('circlefill', 'cross', 'top'), ('cross', 'squarefill', 'topleft'), ('cross', 'cross', 'topleft'), ('circle', 'circle', 'topright'), ('circle', 'circle', 'top'), ('square', 'squarefill', 'topleft'), ('cross', 'square', 'topleft'), ('circle', 'squarefill', 'right'), ('square', 'cross', 'topleft'), ('square', 'circlefill', 'top'), ('squarefill', 'circlefill', 'right'), ('circlefill', 'circle', 'right'), ('cross', 'circlefill', 'right'), ('circlefill', 'squarefill', 'right'), ('cross', 'square', 'topright'), ('square', 'circle', 'topright'), ('cross', 'square', 'right'), ('cross', 'circlefill', 'topleft'), ('cross', 'cross', 'top'), ('circle', 'circlefill', 'top'), ('square', 'square', 'topright'), ('circlefill', 'square', 'right'), ('squarefill', 'cross', 'topleft'), ('cross', 'cross', 'right'), ('cross', 'cross', 'topright'), ('circle', 'squarefill', 'topright'), ('square', 'circle', 'topleft'), ('circlefill', 'circlefill', 'right'), ('circlefill', 'circlefill', 'top'), ('circle', 'square', 'topright'), ('circle', 'cross', 'top'), ('circle', 'circlefill', 'topleft'), ('square', 'squarefill', 'top'), ('squarefill', 'square', 'right'), ('squarefill', 'squarefill', 'top'), ('square', 'circle', 'top'), ('cross', 'square', 'top'), ('circlefill', 'squarefill', 'topleft'), ('circle', 'cross', 'topright'), ('cross', 'circle', 'right'), ('square', 'circlefill', 'right'), ('square', 'cross', 'top'), ('square', 'square', 'right')]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Load dataset")
dataset = RandomRelationalDataset('train', train_combos, transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataset))

speakers = []
target_speakers = []
listeners = []
optim_speakers = []
optim_target_speakers = []
optim_listeners = []
EMAs = []

extractor = Extractor(resnet, f_i)

print("Create models")
for _ in range(population_size):
    speaker = Speaker(msg_length, vocab_size, emb_dim, emb_img, core_state_dim_speaker, f_i).cuda()
    target_speaker = Speaker(msg_length, vocab_size, emb_dim, emb_img, core_state_dim_speaker, f_i).cuda()
    listener = Listener(msg_length, vocab_size, emb_dim, emb_img, core_state_dim_listener, target_proj_dim, core_state_proj_dim, f_i).cuda()
    speakers.append(speaker)
    target_speakers.append(target_speaker)
    listeners.append(listener)
    optim_speakers.append(optim.Adam(speaker.parameters(), lr=lr, betas=betas, eps=eps))
    optim_listeners.append(optim.Adam(listener.parameters(), lr=lr, betas=betas, eps=eps))
    EMAs.append(EMA(target_speaker, kl_EMA))


speakers = np.array(speakers)
target_speakers = np.array(target_speakers)
listeners = np.array(listeners)
optim_speakers = np.array(optim_speakers)
optim_listeners = np.array(optim_listeners)

print("Create losses")
listener_cpc_loss = CPCListenerLoss()
speaker_reinforce_loss = ReinforceSpeakerLoss(speaker_entropy, speaker_kl_target)
imitation_loss = ImitationCrossEntropyLoss(vocab_size)

valid_step = 0
for step in tqdm(range(max_steps)):
    batch = next(iter(train_dataloader))
    combi_batch = batch[0]
    speaker_batch = extractor.extract_multiple(batch[1].cuda())
    listener_batch = extractor.extract_multiple(batch[2].cuda())

    for i in range(len(speakers)):
        optim_speakers[i].zero_grad()
        optim_listeners[i].zero_grad()

    speaker_idx = sample(len(speakers))
    listener_idx = sample(len(listeners))
    
    all_stats_listener = dict()
    all_stats_speaker = dict()
    all_stats_speaker['global_accuracy'] = 0

    for id_speaker, id_listener in zip(speaker_idx, listener_idx):
        speaker = speakers[id_speaker]
        speaker.zero_grad()
        target_speaker = target_speakers[id_speaker]
        target_speaker.zero_grad()
        listener = listeners[id_listener]
        listener.zero_grad()
        optim_speaker = optim_speakers[id_speaker]
        optim_listener = optim_listeners[id_listener]

        action, policy_logits, action_log_prob, entropy, q_values, value = speaker(speaker_batch)
        target_action, target_policy_logits, target_action_log_prob, target_entropy, target_q_values, target_value = target_speaker(speaker_batch, forcing=True, action_to_follow=action)
        listener_predictions, listener_targets = listener(listener_batch, action)

        listener_loss, listener_probs, global_accuracy, reward, listener_stats = listener_cpc_loss.compute_listener_loss(listener_predictions, listener_targets)
        speaker_loss, speaker_stats = speaker_reinforce_loss.compute_speaker_loss(value, action_log_prob, entropy, reward, policy_logits, target_policy_logits)
        
        loss = listener_loss + speaker_loss
        loss.backward()
        optim_speaker.step()
        optim_listener.step()

        stats = dict()
        stats.update(listener_stats)
        stats.update(speaker_stats)
        stats = {k: v/speaker_batch.shape[0] for k, v in stats.items()}

        speaker.avg_score = ((speaker.count_score * speaker.avg_score) / (speaker.count_score + 1)) + (stats['global_accuracy'] / (speaker.count_score+1))
        speaker.count_score += 1
        

        with torch.no_grad():
            EMAs[id_speaker].update(speaker)

    for id_speaker in range(len(speakers)):
        EMA_score(speakers[id_speaker], imit_EMA)

    if (step+1) % imitation_step == 0 and step > 0:
        model_idx = sample(len(speakers), n=n_students+1, replacement=False)
        teacher_student = [(speakers[idx], optim_speakers[idx]) for idx in model_idx]
        teacher_student.sort(key=lambda x: x[0].ema_score)
        students = teacher_student[:-1]
        teacher = teacher_student[-1][0]
        imit_loss = imitation_loss.compute_imitation_loss(teacher, students, speaker_batch)
        stats['imitation_loss'] = imit_loss

    
    for k in stats.keys():
        w.add_scalar(k, stats[k], step+1)

    if (step+1) % save_step == 0 and step > 0:
        for pos, speaker in enumerate(speakers):
            state_dict = dict()
            state_dict['ema_score'] = speaker.ema_score
            state_dict['avg_score'] = speaker.avg_score
            state_dict['count_score'] = speaker.count_score
            torch.save(speaker.state_dict(), os.path.join(models_dir, "speaker_{}_{}".format(pos, step+1)))
            torch.save(state_dict, os.path.join(models_dir, "speaker_var_{}_{}".format(pos, step+1)))
        for pos, listener in enumerate(listeners):
            torch.save(listener.state_dict(), os.path.join(models_dir, "listener{}_{}".format(pos, step+1)))