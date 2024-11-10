import torch  # type: ignore
from torchvision.datasets import Flowers102  # type: ignore
from pathlib import Path  # type: ignore
import sys
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

# Add your project's root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_prototypical import get_transforms, evaluate_on_test
from src.models.prototypical_network import PrototypicalNetwork, compute_prototypes
from src.training.train_prototypical import EpisodeSampler

from umap import UMAP  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
from torchvision.utils import make_grid  # type: ignore


def visualize_episode(
    support_images, support_labels, query_images, query_labels, n_way
):
    """Visualize support and query images from an episode"""
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Support set visualization
    ax1 = plt.subplot(gs[0])
    support_grid = make_grid(support_images, nrow=n_way, normalize=True, padding=2)
    ax1.imshow(support_grid.permute(1, 2, 0).cpu())
    ax1.set_title("Support Set Images")
    ax1.axis("off")

    # Query set visualization
    ax2 = plt.subplot(gs[1])
    query_grid = make_grid(query_images, nrow=n_way, normalize=True, padding=2)
    ax2.imshow(query_grid.permute(1, 2, 0).cpu())
    ax2.set_title("Query Set Images")
    ax2.axis("off")

    plt.tight_layout()
    return fig


def visualize_embeddings(
    support_embeddings,
    query_embeddings,
    support_labels,
    query_labels,
    prototypes,
    method="tsne",
):
    """Visualize embeddings using dimensionality reduction"""
    # Combine all embeddings
    all_embeddings = torch.cat([support_embeddings, query_embeddings, prototypes])
    all_embeddings_np = all_embeddings.detach().cpu().numpy()

    # Prepare labels
    support_labels_np = support_labels.cpu().numpy()
    query_labels_np = query_labels.cpu().numpy()
    prototype_labels = np.arange(len(prototypes))

    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = UMAP(random_state=42)

    embeddings_2d = reducer.fit_transform(all_embeddings_np)

    # Split back into support, query, and prototypes
    n_support = len(support_embeddings)
    n_query = len(query_embeddings)

    support_2d = embeddings_2d[:n_support]
    query_2d = embeddings_2d[n_support : n_support + n_query]
    prototypes_2d = embeddings_2d[-len(prototypes) :]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot support points
    scatter_support = ax.scatter(
        support_2d[:, 0],
        support_2d[:, 1],
        c=support_labels_np,
        marker="o",
        s=100,
        label="Support",
        alpha=0.6,
    )

    # Plot query points
    scatter_query = ax.scatter(
        query_2d[:, 0],
        query_2d[:, 1],
        c=query_labels_np,
        marker="x",
        s=100,
        label="Query",
        alpha=0.6,
    )

    # Plot prototypes
    scatter_protos = ax.scatter(
        prototypes_2d[:, 0],
        prototypes_2d[:, 1],
        c=prototype_labels,
        marker="*",
        s=200,
        edgecolor="black",
        linewidth=1.5,
        label="Prototypes",
    )

    plt.legend()
    plt.title(f"Embeddings Visualization using {method.upper()}")
    return fig


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def evaluate_prototypical(model_path, n_episodes=100):
    """
    Load trained model and evaluate on multiple test configurations.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Get config from checkpoint
    config = checkpoint["config"]

    # Get evaluation transforms
    _, eval_transform = get_transforms()

    # Load test dataset using the same split as during training
    test_dataset = Flowers102(
        root="./data", split="test", transform=eval_transform, download=True
    )

    # Load model architecture with the same feature dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrototypicalNetwork(
        backbone="resnet50", feature_dim=config["feature_dim"]
    ).to(device)

    # Load the model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode

    # Define the configurations to evaluate
    eval_configs = [
        {"n_way": 5, "k_shot": 1},
        {"n_way": 5, "k_shot": 5},
        {"n_way": 10, "k_shot": 1},
        {"n_way": 10, "k_shot": 5},
        {"n_way": 20, "k_shot": 1},
        {"n_way": 20, "k_shot": 5},
        {"n_way": 40, "k_shot": 1},
        {"n_way": 40, "k_shot": 5},
    ]

    # Create lists to store results
    results = {"configs": [], "accuracies": [], "std_devs": []}

    # Evaluate the model on each configuration
    for eval_config in eval_configs:
        n_way = eval_config["n_way"]
        k_shot = eval_config["k_shot"]
        n_query = config.get("n_query", 5)  # Use default n_query or from config

        # Ensure that the number of classes in the test set is sufficient
        unique_classes = set([label for _, label in test_dataset])
        if len(unique_classes) < n_way:
            print(
                f"Not enough classes ({len(unique_classes)}) for {n_way}-way classification."
            )
            continue

        # Create test sampler for current configuration
        test_sampler = EpisodeSampler(
            test_dataset,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
        )

        # Sample one episode for visualization
        episode_data = test_sampler.sample_episode()

        # Get embeddings
        with torch.no_grad():
            support_embeddings = model(episode_data["support_images"].to(device))
            query_embeddings = model(episode_data["query_images"].to(device))
            prototypes = compute_prototypes(
                support_embeddings, episode_data["support_labels"].to(device)
            )

        # Visualize images
        img_fig = visualize_episode(
            episode_data["support_images"],
            episode_data["support_labels"],
            episode_data["query_images"],
            episode_data["query_labels"],
            n_way,
        )

        # Define base visualization directory
        vis_dir = "/home/zrgong/BloomLens/results/visualizations"
        episode_dir = os.path.join(vis_dir, "episodes")
        embedding_dir = os.path.join(vis_dir, "embeddings")
        results_dir = os.path.join(vis_dir, "evaluation_results")

        # Create directories
        ensure_dir(episode_dir)
        ensure_dir(embedding_dir)
        ensure_dir(results_dir)

        # Update save paths with configuration info
        config_str = f"{n_way}way_{k_shot}shot"

        # Save episode visualization
        img_fig.savefig(os.path.join(episode_dir, f"episode_{config_str}.png"))
        plt.close(img_fig)

        # Save embedding visualizations
        tsne_fig = visualize_embeddings(
            support_embeddings,
            query_embeddings,
            episode_data["support_labels"],
            episode_data["query_labels"],
            prototypes,
            method="tsne",
        )
        tsne_fig.savefig(os.path.join(embedding_dir, f"tsne_{config_str}.png"))
        plt.close(tsne_fig)

        umap_fig = visualize_embeddings(
            support_embeddings,
            query_embeddings,
            episode_data["support_labels"],
            episode_data["query_labels"],
            prototypes,
            method="umap",
        )
        umap_fig.savefig(os.path.join(embedding_dir, f"umap_{config_str}.png"))
        plt.close(umap_fig)

        print("\nVisualizations saved under:")
        print(f"- {episode_dir}")
        print(f"- {embedding_dir}")

        # Evaluate on test set
        print(f"\nEvaluating {n_way}-way {k_shot}-shot classification...")
        mean_acc, std_acc = evaluate_on_test(
            model, test_sampler, device, n_episodes=n_episodes
        )

        print(
            f"Test Accuracy for {n_way}-way {k_shot}-shot: {mean_acc:.4f} ± {std_acc:.4f}"
        )

        # Store results
        config_name = f"{n_way}-way {k_shot}-shot"
        results["configs"].append(config_name)
        results["accuracies"].append(mean_acc * 100)  # Convert to percentage
        results["std_devs"].append(std_acc * 100)

    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create bar plot
    bars = plt.bar(
        results["configs"],
        results["accuracies"],
        yerr=results["std_devs"],
        capsize=5,
        color="skyblue",
        alpha=0.8,
    )

    # Customize the plot
    plt.title(
        "Prototypical Network Performance Across Different Configurations",
        fontsize=12,
        pad=20,
    )
    plt.xlabel("Configuration", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prototypical_evaluation_results.png"))
    plt.close()

    print(f"\nEvaluation results visualization saved under {results_dir}")


if __name__ == "__main__":
    model_path = "/home/zrgong/data/BloomLens/checkpoints/run_20241110_093027/stage_3/best_model.pt"
    evaluate_prototypical(model_path, n_episodes=100)
