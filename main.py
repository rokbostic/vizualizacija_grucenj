# python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import sklearn
import scipy


disMatrix = np.array([[0, 23, 27, 20],
                      [23, 0, 30, 28],
                      [27, 30, 0, 30],
                      [20, 28, 30, 0]], dtype=float)

disMatrix2 = np.array([[0, 5, 9, 9, 8],
                       [5, 0, 10, 10, 9],
                       [9, 10, 0, 8, 7],
                       [9, 10, 8, 0, 3],
                       [8, 9, 7, 3, 0]], dtype=float)


def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst] if max_val != min_val else [0] * len(lst)


def printGraph(res):

    for i in range(len(res)):
        for j in [0, 1]:
            print(str(i+len(res)+1) + "->" + str(res[i][j][0]) + ":%0.3f" % res[i][j][1])


def normalize(tree):
    minima = min(map(lambda x: min(x[0][1], x[1][1]), tree))
    maxima = max(map(lambda x: max(x[0][1], x[1][1]), tree))

    for b in tree:
        b[0][1] = (b[0][1] - minima) / (maxima - minima)
        b[1][1] = (b[1][1] - minima) / (maxima - minima)


"""
Clustering

"""


class Clustering:
    def __init__(self, tree):
        self.n = len(tree)+1
        self.tree = tree

        self.cutting_points = []
        for b in self.tree:
            self.cutting_points.append(b[0][1])
            self.cutting_points.append(b[1][1])
        self.cutting_points.sort()

        self.deepest = max(self.cutting_points) # self.cutting_points[-1]
        self.cutting_points = self.cutting_points[1:-1]

    def cluster(self, max_depth):

        labels = [-1] * (self.n + len(self.tree))
        label_last = -1

        def label_clusters(ind):
            labels[ind] = label_last
            ind -= self.n
            if ind >= 0:
                for o in [0, 1]:
                    label_clusters(self.tree[ind][o][0])

        for i in reversed(range(len(self.tree))):
            for o in [0, 1]:

                if labels[i] != -1:
                    continue

                p = self.tree[i][o][0]
                d = self.tree[i][o][1]

                if d > max_depth:
                    label_last += 1
                    label_clusters(p)

        return labels

"""
Joining

"""


def create_linkage(tree):
    n = len(tree)+1

    linkage = []
    for b in tree:
        obs = 0
        d = b[0][1] + b[1][1]

        obs += 1 if b[0][0] < n else linkage[b[0][0] - n][3]
        obs += 1 if b[1][0] < n else linkage[b[1][0] - n][3]

        linkage.append(np.array([b[0][0], b[1][0], d, obs]))

    return np.array(linkage)


def create_tree(linkage):
    tree = []
    for l in linkage:
        tree.append( [ [l[0], l[2]], [l[1], l[2]] ] )
    return tree


def sum_of_depths(tree, current=None, depth=0):
    n = len(tree) + 1

    if current is None:
        current = len(tree) + n - 1

    ind = current - n
    if ind < 0:
        return

    for i in [0, 1]:
        tree[ind][i][1] += depth
        sum_of_depths(tree, tree[ind][i][0], tree[ind][i][1])


def sum_of_depths2(tree):
    n = len(tree) + 1

    for b in tree:
        for i in [0, 1]:
            ind = b[i][0]-n
            b[i][1] += 0 if ind < 0 else tree[ind][0][1] + tree[ind][1][1]


def add_branches(tree, data):
    n = len(tree)+1

    branches = []
    for b in tree:
        p1 = data[int(b[0][0])] if int(b[0][0]) < n else branches[int(b[0][0])-n]
        p2 = data[int(b[1][0])] if int(b[1][0]) < n else branches[int(b[1][0])-n]

        point = np.average( (p1, p2), axis=0)
        # point = point.reshape(1, len(point))
        branches.append(point)
    return np.array(branches)


class NeighborJoining:
    def __init__(self, data):

        self.data = data
        self.n = len(self.data)

        tree_NJ = self.runNeighborJoining()
        normalize(tree_NJ)  # Values can be negative if not normalized
        sum_of_depths2(tree_NJ)

        clusterer = Clustering(tree_NJ)
        cut_point = self.get_cut_point(clusterer)[0]

        labels_NJ = clusterer.cluster(cut_point)

        # Dendograms

        linkage_matrix_NJ = create_linkage(tree_NJ)
        linkage_matrix_ward = scipy.cluster.hierarchy.linkage(self.data, method='ward')

        self.dendogram(linkage_matrix_NJ, "Dendrogram (Neighbor Joining)")
        self.dendogram(linkage_matrix_ward, "Dendrogram (Ward Method)")

        # Scatter plots

        # Neighbor Joining

        branches_NJ = add_branches(tree_NJ, data)  # We add branch points to the data before dimensionality reduction
        self.X = sklearn.manifold.TSNE().fit_transform(np.concatenate((self.data, branches_NJ)))
        self.scatter(self.X, tree_NJ, labels_NJ, "Scatter Plot (Neighbor Joining)")

        # Ward
        tree_ward = create_tree(linkage_matrix_ward)
        labels_ward = np.append(sklearn.cluster.AgglomerativeClustering().fit(self.data).labels_, [-1]*(self.n-1))

        branches_ward = add_branches(tree_ward, data)  # We add branch points to the data before dimensionality reduction
        self.X = sklearn.manifold.TSNE().fit_transform(np.concatenate((self.data, branches_ward)))
        self.scatter(self.X, tree_ward, labels_ward, "Scatter Plot (Ward Method)")

    def get_cut_point(self, clusterer):

        scores = []
        scores2 = []

        for point in clusterer.cutting_points:
            labels = clusterer.cluster(point)

            score = sklearn.metrics.silhouette_score(self.data, labels[:self.n])
            scores.append(score)

            score2 = sklearn.metrics.calinski_harabasz_score(self.data, labels[:self.n])
            scores2.append(score2)

        scores = normalize_list(scores)
        best_cut = scores.index(max(scores))
        best = clusterer.cutting_points[best_cut]

        scores2 = normalize_list(scores2)
        best_cut2 = scores2.index(max(scores2))
        best2 = clusterer.cutting_points[best_cut2]

        df = pd.DataFrame()
        df['x'] = np.array(clusterer.cutting_points)
        df['Silhouette score'] = np.array(scores)
        df["Calinski Harabsz score"] = np.array(scores2)

        fig = px.line(df, x='x', y=["Silhouette score", "Calinski Harabsz score"], title="Comparison of scores")
        fig.update_layout(
            xaxis_title="Cutting point",
            yaxis_title="Score"
        )

        fig.show()

        return best, best2

    def dendogram(self, linkage_matrix, title, line=None):

        # Create the dendrogram
        plt.figure(figsize=(8, 5))
        dendrogram = scipy.cluster.hierarchy.dendrogram(linkage_matrix)

        if line is not None:
            # Draw a horizontal line at a specified height (threshold)
            threshold = line
            plt.axhline(y=threshold, color='r', linestyle='--')

        plt.title(title)
        plt.xlabel("Observation")
        plt.ylabel("Depth")
        plt.show()

    def scatter(self, X, tree, labels, title):
        # Load Iris dataset
        df = pd.DataFrame()
        df['x'] = X.T[0]
        df['y'] = X.T[1]
        df['labels'] = labels
        df['labels'] = df['labels'].astype(str)

        df['index'] = [i for i in range(len(X))]
        df['tree'] = [i >= self.n for i in range(len(X))]

        # Create scatter plot
        fig = px.scatter(df, x='x', y='y',
                         color='labels', title=title,
                         symbol='tree',
                         hover_data={"index": True}
                         )
        fig.update(layout_coloraxis_showscale=False)

        def get_lines():
            n = len(tree)+1
            lines_data = []
            for i in range(len(tree)):
                p0 = i+n
                p1 = tree[i][0][0]
                p2 = tree[i][1][0]

                lines_data.append([p0, p1, 0])
                lines_data.append([p0, p2, 0])
            return lines_data

        lines = get_lines()

        for line_data in lines:
            a, b, d = line_data[0], line_data[1], line_data[2]
            df_line = df.iloc[[a, b]]

            # p = d / self.deepest

            # r_col = "0"
            # g_col = "0"
            # b_col = "0"

            # r_col = str(int(255 * (1-p)))
            # b_col = str(int(255 * p))

            line1 = px.line(df_line, x="x", y="y")
            line1_trace = line1.data[0]
            line1_trace.line.color = "red"  # "rgba("+r_col+", "+g_col+", "+b_col+", 0.5)"  # Blue
            line1_trace.line.width = 0.5
            fig.add_trace(line1_trace)

        fig.update_layout(
            xaxis=dict(fixedrange=False),  # Allow zooming on X-axis
            yaxis=dict(fixedrange=False),  # Allow zooming on Y-axis
            dragmode="pan",  # Enable panning
        )

        fig.show(config={"scrollZoom": True})

    def runNeighborJoining(self):

        D = sklearn.metrics.pairwise_distances(self.data, metric='sqeuclidean')

        n = len(D)

        nodes = [i for i in range(n)]
        last = 0

        tree = [None] * (n - 1)

        def add_merge(f, g, d_fu, d_gu):

            f = nodes[f]
            g = nodes[g]

            tree[last] = [[f, d_fu], [g, d_gu]]


        while True:
            if 2 == n:
                f, g, d = 0, 1, D[0][1]
                add_merge(f, g, d/2, d/2)
                break

            dsum_i = np.sum(D, axis=0)
            dsum_j = dsum_i.reshape((n, 1))

            # 1. Based on the current distance matrix, calculate a matrix Q
            Q = (n - 2) * D - dsum_i - dsum_j
            np.fill_diagonal(Q, 0.)

            # 2. Find the pair of distinct taxa i and j (i.e. with i ≠ j) for which Q(i, j) is smallest.
            index = np.argmin(Q)

            f = index // n
            g = index % n

            # Make a new node that joins the taxa i and j, and connect the new node to the central node.

            # 3. Calculate the distance from each of the taxa in the pair to this new node.

            d_fu = 1/2 * D[f, g] + 1 / (2 * (n - 2)) * (dsum_i[f] - dsum_i[g])
            d_gu = D[f, g] - d_fu

            # 4. Calculate the distance from each of the taxa outside of this pair to the new node.

            d_uk = 1/2 * (D[f, :] + D[g, :] - D[f, g])

            # 5. Start the algorithm again, replacing the pair of joined neighbors with the new node
            # and using the distances calculated in the previous step.

            D = np.insert(D, n, d_uk, axis=0)
            d_uk = np.insert(d_uk, n, 0., axis=0)
            D = np.insert(D, n, d_uk, axis=1)
            D = np.delete(D, [f, g], 0)
            D = np.delete(D, [f, g], 1)

            add_merge(f, g, d_fu, d_gu)

            nodes.append(last+self.n)
            last += 1

            if f > g:
                nodes.pop(f)
                nodes.pop(g)
            else:
                nodes.pop(g)
                nodes.pop(f)

            n -= 1

        return tree


def main():
    iris = sklearn.datasets.load_iris()

    data = iris.data
    # data = sklearn.decomposition.PCA(n_components=2).fit_transform(self.data)
    data_labels = iris.target

    NeighborJoining(data)


if __name__ == "__main__":

    main()




