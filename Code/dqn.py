"""

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=10,
        batch_size=5,
        e_greedy_increment=None,
        memory_size=100,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.hourly_stock_history = []
        self.learn_step_counter = 0
        
        self.memory_counter = 20
        self.memory = np.zeros((memory_size, n_features * 2 + 2))
        
        # Build evaluation and target networks
        self.eval_net = self._build_net('eval_net')
        self.target_net = self._build_net('target_net')
        
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        
    def _build_net(self, name):
        model = keras.Sequential(name=name)
        model.add(layers.Input(shape=(self.n_features,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(self.n_actions, activation='linear'))
        return model
    
    
    
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = np.expand_dims(observation, axis=0)
        
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation).numpy()
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        
        self.hourly_stock_history.append(action)
        return action
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print("\nTarget network parameters replaced\n")

        # Sample memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        states = batch_memory[:, :self.n_features]
        actions = batch_memory[:, self.n_features]
        rewards = batch_memory[:, self.n_features + 1]
        next_states = batch_memory[:, -self.n_features:]

        q_next = self.target_net(next_states)
        q_target = rewards + self.gamma * tf.reduce_max(q_next, axis=1)

        with tf.GradientTape() as tape:
            q_eval = self.eval_net(states)
            a_indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
            q_eval_wrt_a = tf.gather_nd(q_eval, a_indices)
            loss = tf.reduce_mean((q_target - q_eval_wrt_a) ** 2)
        
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        
        # Safely increment epsilon
        if self.epsilon_increment is not None:
            self.epsilon += self.epsilon_increment
            self.epsilon = min(self.epsilon, self.epsilon_max)

        self.learn_step_counter += 1
        
    def get_hourly_stocks(self):
        return self.hourly_stock_history
    
    def reset_hourly_history(self):
        self.hourly_stock_history = []

# Example usage
if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4)