import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hand_env import HandEnv


class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, s_t, s_t1):
        x = torch.cat([s_t, s_t1], dim=-1)
        return self.net(x)
    
    def predict(self, state, next_state):
        self.eval() #evaluation mode for model
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0) # enumerate and convert to tensor
            s_t1 = torch.FloatTensor(next_state).unsqueeze(0)
            action = self.forward(s_t, s_t1).numpy()[0] # remove enumerated values
        self.train()
        return action
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


def collect_transitions(env, n_transitions=10000):
    print(f"\n {n_transitions} transitions are being collected for random actions")
    
    states_list = []
    actions_list = []
    next_states_list = []
    
    transitions_collected = 0
    episode = 0
    
    while transitions_collected < n_transitions:
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 150 and transitions_collected < n_transitions:
            state_before = obs.copy()   # current state
            
            action = env.action_space.sample()  # random ahh action taken

            obs, reward, terminated, truncated, _ = env.step(action)    # take step
            
            state_after = obs.copy()    # next state (after action)
            
            # store

            states_list.append(state_before)
            actions_list.append(action)
            next_states_list.append(state_after)
            
            transitions_collected += 1
            steps += 1
            done = terminated or truncated
            
            time.sleep(0.001)  
        
        episode += 1
        if episode % 10 == 0:
            print(f"Episode {episode}, collected {transitions_collected}/{n_transitions} transitions")
    
    return (
        np.array(states_list, dtype=np.float32),
        np.array(actions_list, dtype=np.float32),
        np.array(next_states_list, dtype=np.float32)
    )


def train_inverse_model(model, states, actions, next_states,n_epochs=50, batch_size=128):
    print(f"\ntraining the inverse model for {n_epochs} number of epochs")
    
    n_samples = len(states) # number of transitions
    n_batches = n_samples // batch_size
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)      # each batch will train on same states but random order
        states_shuffled = states[indices]
        actions_shuffled = actions[indices]
        next_states_shuffled = next_states[indices]
        
        epoch_loss = 0.0        # total addition of loss
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            
            batch_states = states_shuffled[start:end]
            batch_actions = actions_shuffled[start:end]
            batch_next_states = next_states_shuffled[start:end]
            
            s_t = torch.FloatTensor(batch_states)
            a_t = torch.FloatTensor(batch_actions)
            s_t1 = torch.FloatTensor(batch_next_states)
            
            predicted_actions = model(s_t, s_t1)
            
            loss = model.loss_fn(predicted_actions, a_t)
            
            model.optimizer.zero_grad()     # reset graident 
            loss.backward()     # compute graident 
            model.optimizer.step()  # optimisation 
            
            epoch_loss += loss.item()   
        
        avg_loss = epoch_loss / n_batches
        
        if epoch % 5 == 0:
            print(f"epoch {epoch}/{n_epochs} and Loss: {avg_loss:.6f}")
    
    print("Training has been completed")


def test_inverse_model(env, model, n_test_steps=50):
    print("\n------------- testing the inverse model -------------------")
    
    obs, _ = env.reset()
    
    total_error = 0.0
    
    for step in range(n_test_steps):
        state_before = obs.copy()

        actual_action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(actual_action)
        
        state_after = obs.copy()
        
        predicted_action = model.predict(state_before, state_after)
        
        error = np.mean((predicted_action - actual_action) ** 2)
        total_error += error
        
        if step < 5:
            print(f"Step {step}: \n")
            print(f"  Actual action:    {actual_action} \n")
            print(f"  Predicted action: {predicted_action} \n")
            print(f"  MSE: {error:.6f} \n")
        
        time.sleep(0.02)
        
        if done or trunc:
            break
    
    avg_error = total_error / n_test_steps
    print(f"\nMSE error: {avg_error:.6f}")


def predict_demo_actions(model, demo_states):
    predicted_actions = []
    
    for i in range(len(demo_states) - 1):
        action = model.predict(demo_states[i], demo_states[i+1])
        predicted_actions.append(action)
    
    return np.array(predicted_actions)


if __name__ == "__main__":
    print("\n------------- training the inverse model -------------------")
    
    env = HandEnv(render=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    model = InverseModel(state_dim, action_dim)
    
    states, actions, next_states = collect_transitions(env, n_transitions=5000)
    
    print(f"Defined | state dim: {states.shape} | action dim: {actions.shape} | next state dim: {next_states.shape}")
    train_inverse_model(model, states, actions, next_states, n_epochs=50, batch_size=128)
    
    test_inverse_model(env, model, n_test_steps=20)
    
    model.save('inverse_model_trained.pth')
    print("\nModel has been trained and saved'")
    
    env.close()
    print("\n------------------ Training Complete --------------------------")