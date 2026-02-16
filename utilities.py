
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import os
from scipy.stats import bernoulli
from scipy.ndimage import gaussian_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def trend(x, method="Lowess", bandwidth=0.2, span=0.2):
    """
    Adaopted from T.Bury's ewstools library
    Detrend the time series using a chosen method.

    Parameters
    ----------
    x: time series
    method : str, optional
        Method of detrending to use.
        Select from ['Gaussian', 'Lowess']
        The default is 'Gaussian'.
    bandwidth : float, optional
        Bandwidth of Gaussian kernel. Provide as a proportion of data length
        or as a number of data points. As in the R function ksmooth
        (used by the earlywarnings package in R), we define the bandwidth
        such that the kernel has its quartiles at +/- 0.25*bandwidth.
        The default is 0.2.
    span : float, optional
        Span of time-series data used for Lowess filtering. Provide as a
        proportion of data length or as a number of data points.
        The default is 0.2.

    Returns
    -------
    Trend

    """
    n = x.shape[0]
    if method == "Gaussian":
        # Get size of bandwidth in terms of num. datapoints if given as a proportion
        if 0 < bandwidth <= 1:
            bw_num = bandwidth * n
        else:
            bw_num = bandwidth

        # Use the gaussian_filter function provided by Scipy
        # Standard deviation of kernel given bandwidth
        # We want about quartiles to fall +-0.25 * bandwidth
        sigma = (0.25 / 0.675) * bw_num
        return gaussian_filter(x, sigma=sigma, mode="reflect")

    if method == "Lowess":
        # Convert span to a proportion of the length of the data
        if not 0 < span <= 1:
            span_prop = span / n
        else:
            span_prop = span

        return lowess(
            x, range(1, n+1), frac=span_prop, is_sorted=True
        )[:, 1]
    

def read_X_y_data(train_test_files, class_labels = [0, 1], detrend = True, **kwargs):
    """
    Load training and testing datasets for multiple classes, optionally detrend
    each time series, and return PyTorch tensors.

    This function expects one training file and one testing file per class.
    Each file should contain samples arranged row-wise (one sample per row).

    Parameters
    ----------
    train_test_files : list of tuple[str, str]
        A list where each element corresponds to a class and contains:
            (train_filename, test_filename)

        The expected format is:

            [
                (class_0_train_filename, class_0_test_filename),
                (class_1_train_filename, class_1_test_filename),
                ...
                (class_n_train_filename, class_n_test_filename)
            ]

        Files are assumed to be located in "./data/" and are loaded using
        `numpy.loadtxt`.

        IMPORTANT: The position in the list must match the class label index,
        i.e. `train_test_files[label]` must correspond to `label`.

    class_labels : list of int, default=[0, 1]
        Integer class labels to load. These must correspond to valid indices
        in `train_test_files`.

        The order of labels determines the one-hot encoding. If using a
        non-tipping class, it should typically be the first class (label 0) so 
        that models can learn probabilities close to 0 for the non-tipping class.

    detrend : bool, default=True
        If True, each sample (row) in both training and testing sets is
        detrended by subtracting a trend computed using the `trend` function.
        If the data are already detrended, set this to False.

    **kwargs
        Additional keyword arguments passed to the `trend` function.

    Returns
    -------
    X_train : torch.Tensor
        Training input data of shape (n_train_samples, n_features),
        dtype=torch.float32.

    y_train : torch.Tensor
        One-hot encoded training labels of shape
        (n_train_samples, n_classes), dtype=torch.float32.

    X_test : torch.Tensor
        Testing input data of shape (n_test_samples, n_features),
        dtype=torch.float32.

    y_test : torch.Tensor
        One-hot encoded testing labels of shape
        (n_test_samples, n_classes), dtype=torch.float32.

    Notes
    -----
    - Training data are shuffled (three times) before conversion to tensors.
    - Data are stacked across classes using `numpy.vstack`.
    - All returned tensors are moved to the global `device`.

    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for label in class_labels:
        train_filename, test_filename = train_test_files[label]
        train_filename = os.path.join("./data", train_filename)
        test_filename = os.path.join("./data", test_filename)
        X_train.append(np.loadtxt(train_filename))
        X_test.append(np.loadtxt(test_filename))
        
        class_vector = np.zeros(len(class_labels))
        class_vector[label] = 1
        y_train.extend([class_vector]*X_train[-1].shape[0])
        y_test.extend([class_vector]*X_test[-1].shape[0])

    idx = np.arange(len(y_train), dtype=np.int8)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)

    X_train = X_train[idx]
    y_train = y_train[idx]

    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    if detrend:
        T = np.apply_along_axis(trend, 1, X_train, **kwargs)
        X_train -= T
        T = np.apply_along_axis(trend, 1, X_test, **kwargs)
        X_test -= T
        # X = (X - np.mean(X, axis=1).reshape(-1, 1))/np.std(X, axis = 1).reshape(-1, 1)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)


    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, y_train, X_test, y_test

    

def prepare_train_test_data(X_train, y_train,
                            X_test, y_test,
                            num_batches = 10,
                            shorter_sequences = False,
                            min_short_seq_len = 50,
                            max_short_seq_len = 500,
                            device = device):
    """
    Prepare training and testing tensors for model input, optionally generate
    variable-length training sequences, and split the training set into batches.

    This function:
    1. Optionally shortens a subset of training sequences to simulate
       variable-length inputs.
    2. Pads shortened sequences (if enabled).
    3. Adds a channel dimension required by CNN/RNN models.
    4. Splits the training data into `num_batches` batches.

    Parameters
    ----------
    X_train : torch.Tensor
        Training input tensor of shape (n_train_samples, sequence_length).

    y_train : torch.Tensor
        Training labels of shape (n_train_samples, n_classes).

    X_test : torch.Tensor
        Test input tensor of shape (n_test_samples, sequence_length).

    y_test : torch.Tensor
        Test labels of shape (n_test_samples, n_classes).

    num_batches : int, default=10
        Number of batches to divide the training data into.
        Batches are created by splitting the dataset into batches. 
        The final batch may contain more samples if the division is not exact.

    shorter_sequences : bool, default=False
        If True, randomly shortens approximately 30% of training sequences
        (with probability 0.3) to simulate variable-length inputs.
        Shortened sequences are left-padded using `pad_sequence`. This important 
        to improve skill for shorter sequences.

    min_short_seq_len : int, default=50
        Minimum length of randomly shortened sequences.

    max_short_seq_len : int, default=500
        Maximum length of randomly shortened sequences.

    device : torch.device
        Device to which tensors are moved (e.g., "mps", "cpu" or "cuda").

    Returns
    -------
    X_train_batch : list of torch.Tensor
        List of length `num_batches`. Each element is a batch tensor of shape:
            (batch_size_i, 1, sequence_length_padded)

        The additional dimension (size 1) corresponds to the input channel
        required by CNN/RNN models.

    y_train_batch : list of torch.Tensor
        List of label tensors corresponding to each training batch.
        Each tensor has shape (batch_size_i, n_classes).

    X_test : torch.Tensor
        Test inputs reshaped to (n_test_samples, 1, sequence_length)
        and moved to `device`.

    y_test : torch.Tensor
        Test labels (unchanged shape), returned for convenience.

    Notes
    -----
    - When `shorter_sequences=True`, sequences are randomly truncated by
      selecting a random starting index and random length within the
      specified bounds.
    - Padding is applied on the left (`padding_side='left'`).
    - A singleton channel dimension is added using `unsqueeze(dim=1)`
      so that the final input shape matches:
          (batch_size, input_channels=1, sequence_length).
    - Training batches are created by contiguous slicing, not random
      shuffling.
    """

    seq_len = X_train[0].shape[-1]
    n_train = y_train.shape[0] 

    # add shorter sequences so the classifer can deal with sequences of variable lengths
    if shorter_sequences:
        c = bernoulli.rvs(p=0.7, size=n_train)
        X_train_short = []
        for i in range(n_train):
            x = X_train[i]
            if c[i] == 1:
                X_train_short.append(x)
            else:
                l = random.randint(min_short_seq_len, max_short_seq_len)
                start = random.randint(0, seq_len-1-l) # give at least l time steps
                X_train_short.append(x[start:start+l])

        X_train = pad_sequence(X_train_short, batch_first = True, padding_side='left')

    # our RNN classifier expects inputs of the shape (batch_size, input_size, sequence length)
    # for the CNN to process but X_train_pad is currently of the shape (batch_size, sequence length)
    # unsqueeze(dim = d) adds an extra dimension next to d but this is equal to 1
    # One can also use reshape or view functions in pytorch
    X_train = X_train.unsqueeze(dim = 1).to(device)
    X_test = X_test.unsqueeze(dim = 1).to(device)

    batch_size = n_train//num_batches
    idx = np.arange(num_batches)*batch_size
    idx = np.r_[idx, n_train]

    X_train_batch = [X_train[idx[i]:idx[i+1]] for i in range(num_batches)]
    y_train_batch = [y_train[idx[i]:idx[i+1]] for i in range(num_batches)]

    return X_train_batch, y_train_batch, X_test, y_test

def show_sequence_as_a_tape(x, start_at = 1):
    """
    Convert a single sequence into a causal "tape" representation.

    This function creates a stacked tensor containing all progressive
    subsequences of the time series:

        [x[:start_at],
         x[:start_at+1],
         ...
         x[:T]]

    where T = x.shape[0].

    Each subsequence is left-padded so that all rows have equal length, forming
    a 2D tensor. A singleton channel dimension is then added to match the
    expected input shape of CNN/RNN models.

    Parameters
    ----------
    x : torch.Tensor
        One-dimensional time series of shape (T,),
        where T is the sequence length.

    start_at : int, default=1
        Index at which to begin generating subsequences.
        - If start_at=1, subsequence starts from x[:1].
        - Must satisfy 1 <= start_at <= T.

    Returns
    -------
    tape : torch.Tensor
        Tensor of shape (T - start_at + 1, 1, T), where:

        - Dimension 0: time steps (progressive subsequences)
        - Dimension 1: input channel (size 1)
        - Dimension 2: padded sequence length (T)

        Padding is applied on the left (`padding_side='left'`).

    Notes
    -----
    - This representation is useful for visualising or evaluating
      sequence models in a causal manner (i.e., predictions as more
      time steps are revealed).
    - Internally uses `torch.nn.utils.rnn.pad_sequence`.
    """
    return pad_sequence(
      [x[:i] for i in range(start_at, x.shape[0]+1)], batch_first=True, padding_side='left'
      ).unsqueeze(dim = 1) 

# model and training
def train(model, X_train, y_train, X_test, y_test, 
          loss_fn, optimizer, train_losses, test_losses, n_epochs = 100, print_every = 20):
    """
    Train a model using pre-batched training data and evaluate on a fixed test set.

    This function performs mini-batch training over multiple epochs.
    At each epoch:
        - The model is updated batch-by-batch on the training data.
        - The test loss is evaluated (without gradients) after each batch.
        - Average training and test losses are recorded.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.

    X_train : list of torch.Tensor
        List of training input batches.
        Each element must have shape:
            (batch_size_i, input_channels, sequence_length)

    y_train : list of torch.Tensor
        List of training label batches corresponding to `X_train`.
        Each element must have shape:
            (batch_size_i, n_classes)

    X_test : torch.Tensor
        Full test input tensor of shape:
            (n_test_samples, input_channels, sequence_length)

    y_test : torch.Tensor
        Full test label tensor of shape:
            (n_test_samples, n_classes)

    loss_fn : callable
        Loss function (e.g., `torch.nn.CrossEntropyLoss`,
        `torch.nn.BCELoss`, etc.).

    optimizer : torch.optim.Optimizer
        Optimizer instance used to update model parameters.

    train_losses : list
        List to which the average training loss per epoch will be appended.
        Modified in-place.

    test_losses : list
        List to which the average test loss per epoch will be appended.
        Modified in-place.

    n_epochs : int, default=100
        Number of training epochs.

    print_every : int, default=20
        Frequency (in epochs) at which training progress is printed.

    Returns
    -------
    train_losses : list of float
        Average training loss per epoch.

    test_losses : list of float
        Average test loss per epoch.

    Notes
    -----
    - Training data are assumed to already be batched.
    - Test loss is computed without gradient tracking (`torch.no_grad()`).
    - Loss values are averaged over the number of training batches.
    - The function calls `torch.mps.empty_cache()` at the end of each epoch
      (relevant when using Apple Silicon GPUs).
    - The model is set to training mode via `model.train()` at the start.
    """
    model.train()
    N = len(X_train)
    X_test, y_test = X_test.to(device), y_test.to(device)
    for epoch in range(n_epochs):
        # Training
        train_loss_sum = 0
        test_loss_sum = 0
        for (X, y) in zip(X_train, y_train):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad() 
            y_pred_train = model(X)
            train_loss = loss_fn(y_pred_train, y)
            train_loss.backward()
            optimizer.step()
            # Testing
            with torch.no_grad(): # turn of gradients as there is not use for them
                y_pred_test = model(X_test)
                test_loss_sum += loss_fn(y_pred_test, y_test).detach().item()
                train_loss_sum += train_loss.detach().item()
        # average out to get train and loss
        train_loss = train_loss_sum/N
        test_loss = test_loss_sum/N
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if device == 'mps':
            torch.mps.empty_cache()

    return train_losses, test_losses


# plotting
def plot_train_and_test_loss(train_loss, test_loss):
    """
    takes in a list of train and test lossess and plots them in one panel
    """
    plt.plot(range(1, len(train_loss)+1), train_loss, label = "training loss")
    plt.plot(range(1, len(train_loss)+1), test_loss, label = "test loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

def plot_model_on_test(model, X_test, y_test, n_plots = 30, plot_mean = True):
    """
    Plot the model's predicted tipping probability over time on test data.

    This function evaluates the model in a causal manner: for each selected
    trajectory, predictions are computed on progressively longer subsequences
    of the sequence (using `show_sequence_as_a_tape`). This produces a
    time-evolving probability curve for the tipping class.

    The function can either:
        - Plot individual probability trajectories, or
        - Plot the mean ± 1 standard deviation across multiple trajectories.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model. The model must output class probabilities
        or logits with shape (batch_size, n_classes), where column index 1
        corresponds to the tipping class probability.

    X_test : torch.Tensor
        Test inputs of shape:
            (n_samples, 1, sequence_length)

    y_test : torch.Tensor
        One-hot encoded test labels of shape:
            (n_samples, n_classes)

    n_plots : int or None, default=None
        Number of tipping and non-tipping trajectories to plot.

    plot_mean : bool, default=True
        If False:
            Plot individual probability trajectories.
        If True:
            Plot the mean tipping probability across selected trajectories,
            with shaded regions corresponding to ±1 standard deviation.

    Returns
    -------
    None
        Displays a matplotlib figure.

    Notes
    -----
    - Tipping and non-tipping samples are selected using `y_test[:, 1]`.
    - For each selected trajectory:
          model(show_sequence_as_a_tape(sequence))
      produces a probability curve over time.
    - When `plot_mean=True`, the function:
          1. Collects all probability curves,
          2. Computes mean and standard deviation across samples,
          3. Plots mean curves with shaded uncertainty bands.
    - `torch.mps.empty_cache()` is called during iteration (relevant for
      Apple Silicon GPU memory management).
    - The function assumes binary classification with the tipping class
      at index 1.
    """
    plt.figure(figsize=(8, 5))
    tip_cond = y_test[:, 1]== 1
    non_tip_cond = y_test[:, 1] == 0
    X_test_tip = X_test[tip_cond]
    X_test_non_tip = X_test[non_tip_cond]
    if n_plots is None:
        idx = range(min(tip_cond.sum().cpu(), non_tip_cond.sum().cpu()))
    else:
        idx = np.random.randint(0, min(tip_cond.sum().cpu(), non_tip_cond.sum().cpu()), n_plots)
    x = np.arange(1, X_test[0].shape[-1]+1)
    probs_non_tip = []
    probs_tip = []
    for j in range(len(idx)):
        i = idx[j]
    # for i in range(min(X_test_tip.shape[0], X_test_non_tip.shape[0])):
        x0_tip = show_sequence_as_a_tape(X_test_tip[i, 0, :])
        x0_non_tip = show_sequence_as_a_tape(X_test_non_tip[i, 0, :])
        probs_tip.append(model(x0_tip)[:, 1].detach().cpu().numpy())
        probs_non_tip.append(model(x0_non_tip)[:, 1].detach().cpu().numpy())
        if not plot_mean:
            if j == 0:
                plt.plot(x, probs_tip[-1], label = "Tipping", color = "red")
                plt.plot(x, probs_non_tip[-1], label = "Non-tipping", color = "blue")
            else:
                plt.plot(x, probs_tip[-1], color = "red")
                plt.plot(x, probs_non_tip[-1], color = "blue")
        if device == 'mps':
            torch.mps.empty_cache()

    if plot_mean:
        # convert to numpy
        probs_tip = np.array(probs_tip)
        probs_non_tip = np.array(probs_non_tip)

        # mean and standard deviation of those that would tip
        mean_tip = np.mean(probs_tip, axis=0)
        std_tip  = np.std(probs_tip, axis=0)

        # mean and standard deviation of those that would not tip
        mean_non = np.mean(probs_non_tip, axis=0)
        std_non  = np.std(probs_non_tip, axis=0)

        plt.figure(figsize=(8, 5))
        # plot mean
        plt.plot(x, mean_tip, color="red", label="Tipping")
        plt.plot(x, mean_non, color="blue", label="Non-tipping")
        # plot + or - one standard deviations
        plt.fill_between(
            x,
            mean_tip - std_tip,
            mean_tip + std_tip,
            color="red",
            alpha=0.25,
        )
        plt.fill_between(
            x,
            mean_non -  std_non,
            mean_non + std_non,
            color="blue",
            alpha=0.25,
        )

    plt.xlabel("Time step")
    # plt.title(f"Mean tipping probability ± 1 standard deviations")
    plt.title(f"Tipping probability for an ensemble of {len(idx)} tipping and non-tipping trajectories")
    plt.ylabel("Tipping probability")
    plt.legend()
    plt.tight_layout()
    plt.show()