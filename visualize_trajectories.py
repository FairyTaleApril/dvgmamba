import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.pairwise import cosine_distances

matplotlib.use('TkAgg')


def read_trajectory(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines if not line.startswith('#')]
    return np.array(data)[:, :3] if data else None


def compute_diversity(traj_list):
    endpoints = [traj[-1] for traj in traj_list if traj.shape[0] > 1]
    if len(endpoints) < 2:
        return 0.0
    dist_matrix = cosine_distances(endpoints)
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    return np.mean(upper)


def compute_smoothness(traj):
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    angles = []
    for i in range(1, len(diffs)):
        cos_sim = np.dot(diffs[i-1], diffs[i]) / (norms[i-1]*norms[i] + 1e-8)
        angles.append(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    return np.mean(angles) if angles else 0.0


def compute_avg_smoothness(traj_list):
    return np.mean([compute_smoothness(traj) for traj in traj_list if traj.shape[0] > 2])


def compute_volume(traj_list):
    all_points = np.concatenate(traj_list, axis=0)
    min_xyz = np.min(all_points, axis=0)
    max_xyz = np.max(all_points, axis=0)
    return np.prod(max_xyz - min_xyz)


def analyze_and_plot(ax, folder_path, title):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(files))
    trajs = []

    for i, fname in enumerate(files):
        fpath = os.path.join(folder_path, fname)
        traj = read_trajectory(fpath)
        if traj is None:
            continue
        shifted = traj - traj[0]
        trajs.append(shifted)
        ax.plot(shifted[:, 0], shifted[:, 1], shifted[:, 2], color=cmap(i), alpha=0.7, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    diversity = compute_diversity(trajs)
    smoothness = compute_avg_smoothness(trajs)
    volume = compute_volume(trajs)
    print(f"--- {title} ---")
    print(f"Trajectories loaded: {len(trajs)}")
    print(f"1. Diversity (avg cosine distance): {diversity:.4f}")
    print(f"2. Smoothness (avg turning angle, rad): {smoothness:.4f}")
    print(f"3. Spatial volume used: {volume:.2f}")


def plot_two_folders(folder1, folder2, title1='Mamba', title2='GPT'):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Mamba vs GPT Trajectories", fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    analyze_and_plot(ax1, folder1, title1)
    analyze_and_plot(ax2, folder2, title2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_two_folders('Mamba', 'GPT')
