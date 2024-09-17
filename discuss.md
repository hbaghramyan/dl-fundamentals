05/08/2024
1. 

class MyDataset(Dataset):

    def __init__(self, X, y):
        super().__init__() - THERE IS NO CONSTRUCTOR IN DATASET
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

2. plt.legend(loc="upper left") instead of plt.legend(loc=2)

3. you can avoid model.train() because there are no layers that perform diffently at training and evaluation stages (like dropout and BatchNorm)

4. you do not need model = model.eval()

5. plot_boundary


26/08/24

1. Best coding practice - do not specify variable you don't use


all_x = []
for x, _ in train_loader:
    all_x.append(x)
    
train_std = torch.concat(all_x).std(dim=0)
train_mean = torch.concat(all_x).mean(dim=0)

2. explain why the formula has this concrete form

![alt text](image.png)


$H(p, q) = -\sum_i p_i \log q_i = -y \log \hat{y} - (1 - y) \log(1 - \hat{y}).$


https://en.wikipedia.org/wiki/Cross-entropy

https://en.wikipedia.org/wiki/Entropy_(information_theory)

3. Do not remember the functions and parameters - read the documentation

4. plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64], padding=4, pad_value=0.5, normalize=True), (1, 2, 0)))

16/09/24

1. [__name__ == "__main__"](https://builtin.com/articles/name-python#:~:text=%E2%80%9CIf%20__name__%3D%3D%20',main%20method%20should%20be%20executed.&text=If%20you%20are%20new%20to,or%20without%20a%20main%20method.) in python

2. Implications with multiprocessing

    Let’s break down the behavior of Python's multiprocessing on macOS and Windows more clearly, focusing on why not using the `if __name__ == "__main__":` block can lead to problems.

    ### Understanding Multiprocessing and Script Execution

    When using multiprocessing in Python (like when you set `num_workers > 0` in a `DataLoader`), the operating system needs to create new processes to handle these additional tasks.

    1. **Linux (Fork Method):**
    - On Linux, the new processes are created using the fork method.
    - This method duplicates the parent process, including its current state (variables, code execution point, etc.).
    - No re-importing of the script occurs, so the new process doesn't execute the script from the start.

    2. **macOS and Windows (Spawn Method):**
    - On macOS and Windows, the default method for creating new processes is spawn.
    - This method starts a brand new, fresh Python interpreter process.
    - The new process needs to know what code to run, so it **re-imports the script** that created it.

    ### Why the `if __name__ == "__main__":` Block is Crucial

    When the new interpreter spawns (on macOS/Windows), it starts by re-importing your script to understand what it should execute. Here’s what happens step-by-step if your script doesn't use `if __name__ == "__main__":`:

    1. **Script Import:** When the new process starts, it imports your script, running from the top.
    
    2. **Uncontrolled Execution:** If the DataLoader and dataset creation code is outside of `if __name__ == "__main__":`, it runs as soon as the script is imported, not just when intended.

    3. **Recursive Spawning:** Because the DataLoader with `num_workers > 0` creates new processes, those processes will also re-import and re-execute the script from the top, trying to create their own DataLoader instances.

    4. **Infinite Loop:** This leads to a chain reaction where every new process spawns more processes, which again import and execute the script, trying to spawn even more processes. This results in infinite recursion of process creation.

    ### How the `if __name__ == "__main__":` Block Prevents This

    - The `if __name__ == "__main__":` block ensures that certain code only runs when the script is executed directly, not when it is imported.
    - When a new process spawns and imports the script, it does not execute the code inside `if __name__ == "__main__":`, preventing the recursive spawning issue.
    
    For example:

    ```python
    if __name__ == "__main__":
        # This code will only run when the script is executed directly.
        train_loader = DataLoader(...)
    ```

    In a new process, this block is ignored because `__name__` is not `"__main__"` (it’s the module name). This prevents unintended and repeated execution of the DataLoader creation code, keeping the multiprocessing controlled and functional.

    ### Key Point

    The `if __name__ == "__main__":` block ensures that multiprocessing doesn’t lead to runaway process creation on platforms that use the spawn method, making your script work correctly and efficiently on all operating systems.

3. 

The `transforms.Normalize((0.5,), (0.5,))` step is included to normalize your image data so that its pixel values fall within the \([-1, 1]\) range instead of the default \([0, 1]\) range produced by `transforms.ToTensor()`. Here's why this is beneficial:

* **Normalization Process**: The `transforms.Normalize(mean, std)` function adjusts the pixel values using the formula:

   \[
   \text{output} = \frac{\text{input} - \text{mean}}{\text{std}}
   \]

   By setting `mean` to \((0.5,)\) and `std` to \((0.5,)\), the transformation becomes:

   \[
   \text{output} = \frac{\text{input} - 0.5}{0.5} = 2 \times \text{input} - 1
   \]

   This maps the input pixel values from the \([0, 1]\) range to \([-1, 1]\).

* **Benefits of \([-1, 1]\) Range**:
   - **Neural Network Performance**: Many neural network architectures, especially those using activation functions like `tanh`, perform better when inputs are centered around zero. This can lead to faster convergence during training.
   - **Stability**: Normalizing data can improve numerical stability and make the training process less sensitive to the scale of input features.
   - **Consistency**: If pre-trained models (e.g., those trained on ImageNet) expect inputs in a specific range, normalizing your data accordingly ensures compatibility.

* **Channel-wise Normalization**: The `(0.5,)` tuple indicates that this normalization is applied to each channel individually. For grayscale images (single-channel), this is straightforward. For RGB images, you would provide a mean and standard deviation for each channel.

**In summary**, the `transforms.Normalize((0.5,), (0.5,))` step scales your image data to a \([-1, 1]\) range, which is often preferred for training neural networks due to improved performance and stability.

**Answer:**

Because it scales image pixels from [0, 1] to [–1, 1]; using Normalize((0.5,), (0.5,)) centers and scales the data so neural networks train better with inputs in the [–1, 1] range


