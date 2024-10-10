import torch
from torch.nn.functional import normalize, softmax, kl_div, log_softmax, one_hot, cross_entropy

def cosine_loss(x, y):
    normed_x = normalize(x, p=2., dim=-1)
    normed_y = normalize(y, p=2., dim=-1)
    
    return torch.sum((normed_x - normed_y)**2, dim=-1)

class ReinforceSpeakerLoss():
    def __init__(self, speaker_entropy, speaker_kl_target) -> None:
        self.speaker_entropy = speaker_entropy
        self.speaker_kl_target = speaker_kl_target

    def compute_speaker_loss(self, value, action_log_prob, entropy, reward, policy_logits, target_policy_logits):
        value = torch.permute(value, [2,0,1])
        value = torch.squeeze(value, dim=-1)

        action_log_prob = torch.permute(action_log_prob, [1,0])
        
        entropy = torch.permute(entropy, [1,0])

        # Policy loss via Reinforce
        sg_value = value.detach()
        policy_loss = -torch.mean((reward - sg_value) * action_log_prob, dim=0) # L_pi(theta)
        policy_loss = torch.sum(policy_loss, dim=0)

        entropy = torch.sum(torch.mean(entropy, dim=0), dim=0)
        entropy_loss = -entropy * self.speaker_entropy

        value_loss = torch.mean(torch.square(reward - value), dim=0) # L_V(theta)
        value_loss = torch.sum(value_loss, dim=0)
        value_stats = torch.sum(torch.mean(value, dim=0), dim=0)
        
        policy_logits = softmax(policy_logits, dim=1)
        target_policy_logits = softmax(target_policy_logits, dim=1)       

        batch_size = policy_logits.shape[0]
        msg_length = policy_logits.shape[-1]
        total = 0
        for x in range(batch_size):
            x_mean = 0
            for t in range(msg_length):
                x_t_pair = torch.sum(policy_logits[x,:,t] * torch.log(policy_logits[x,:,t]/target_policy_logits[x,:,t]))
                x_mean += x_t_pair
            x_mean = x_mean / msg_length
            total += x_mean
        
        kl_target_loss = total * self.speaker_kl_target
        
        speaker_loss = policy_loss + entropy_loss + value_loss + kl_target_loss

        stats = dict(
            value=value_stats.detach().item(),
            value_loss=value_loss.detach().item(),
            speaker_loss=speaker_loss.detach().item(),
            policy_loss=policy_loss.detach().item(),
            entropy_loss=entropy_loss.detach().item(),
            kl_target_loss=kl_target_loss.detach().item(),
            speaker_entropy=entropy.detach().item(),
        )
        
        return speaker_loss, stats

class CPCListenerLoss():
    def __init__(self) -> None:
        pass

    def compute_listener_loss(self, predictions, targets):
        batch_size = targets.shape[0]
        feature_dim = targets.shape[1]

        num_distractor = -1

        batch_indices = torch.arange(batch_size).cuda()
        
        cosine_sim = cosine_loss(predictions[:,None,:], targets[None,:,:])

        listener_probs = softmax(cosine_sim, dim=-1)
        
        listener_loss = -log_softmax(cosine_sim, dim=-1)[batch_indices, batch_indices]
        
        accuracy = (torch.argmax(cosine_sim, dim=-1) == batch_indices)
        reward = accuracy * 1
        reward = reward.detach()

        listener_loss = torch.sum(listener_loss, dim=0)

        global_accuracy = torch.sum(reward, dim=0)

        stats = {
            'listener_loss': listener_loss.detach().item(),
            'global_accuracy': global_accuracy.detach().item()
        }

        return listener_loss, listener_probs, global_accuracy, reward, stats

class CPCListenerLossEval():
    def __init__(self) -> None:
        pass

    def compute_listener_loss(self, predictions, targets, original_batch_size=20):
        batch = predictions.shape[0]

        batch_indices = torch.arange(original_batch_size).cuda()

        cosine_sim = cosine_loss(predictions[:,None,:], targets[None,:,:])

        chunks = cosine_sim.unfold(0, original_batch_size, original_batch_size).unfold(1, original_batch_size, original_batch_size)

        global_accuracy = 0

        for i in range(int(batch/original_batch_size)):
            accuracy = (torch.argmax(chunks[i,i], dim=-1) == batch_indices)
            reward = accuracy * 1
            reward = reward.detach()
            global_accuracy = global_accuracy + torch.sum(reward, dim=0)


        listener_loss = 0
        listener_probs = 0
        stats = {
            'listener_loss': listener_loss,#.detach().item(),
            'global_accuracy': global_accuracy.detach().item()
        }

        return listener_loss, listener_probs, global_accuracy, reward, stats

class ImitationCrossEntropyLoss():
    def __init__(self, voc_size):
        self.voc_size = voc_size
    
    def compute_imitation_loss(self, teacher, students, batch):
        with torch.no_grad():
            teacher_action, _, _, _, _, _ = teacher(batch)
            labels = one_hot(teacher_action, num_classes=self.voc_size)
        
        for student, optim in students:
            optim.zero_grad()
            
            action, policy_logits, action_log_prob, entropy, q_values, value = student(batch)
            policy_logits = policy_logits.permute([0,2,1])
            
            logits = log_softmax(policy_logits, dim=-1)
            loss = labels * logits
            loss = -torch.sum(loss, dim=-1)
            loss = torch.sum(torch.mean(loss, dim=-1), dim=0) / logits.shape[0]

            loss.backward()
            optim.step()

        return loss