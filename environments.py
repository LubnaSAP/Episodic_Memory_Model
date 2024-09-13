#!/usr/bin/python
# -*- coding: utf-8 -*-
import gym
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
import holoviews as hv
from utils import create_tensor, row_norm
from visualization import (
    text_font_size,
    node_size,
    edge_size,
    color_index_edge,
    cmap_edge,
    remove_axes,
)
from utils import (
    pos_dict,
)
from networkx.drawing.nx_agraph import graphviz_layout

hv.extension("bokeh", "matplotlib")


class GraphEnv(gym.Env):
    """
    Simple openai-gym environment wrapper.
    Instructions: initialize then set the reward function with goal_func.
    May want to customize __init__, _access_matrix, _node_info,...
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, viz_scale=1.0):
        self.__name__ = "generic-env"
        self.__type__ = "graph"
        self.goal_absorb = False
        self.stay_actions = False
        self.viz_scale = viz_scale
        self._state_information()
        self._transition_information()
        self.set_viz_scheme()
        self.fname_graph = "figures/graph.png"
        self.n_dim = 2  # assume 2D embedding

        # action space defined as number of states
        self.action_space = spaces.Discrete(self.n_state)
        self.naction_max = (self.A != 0).sum(axis=1).max()

        # discrete set of states
        self.observation_space = spaces.Discrete(self.n_state)

        # initialize
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.start
        self.lastaction = None
        return self.state

    def step(self, action):
        if self.A[self.state, action]:
            # available transition
            reward = self.R[self.state, action]
            self.state = action
            if self.state == self.goal:
                done = True
            else:
                done = False
            return self.state, reward, done, {}
        else:
            done = False
            reward = -self.stepcost
            return self.state, reward, done, {}

    def congruent(self, env):
        """
        FUNCTION: Checks if env is a congruent environment to self.
        True if underlying graphs are isomorphic.
        NOTES: More checks in future.
        """
        # return nx.is_isomorphic(self.G, env.G)
        return np.all(self.A == env.A)

    def analogous_policy(self, T, fill_prob=0.001):
        """
        FUNCTION: Returns the policy closest to T that is compatible with the current state-space structure (if possible).
        INPUTS: T = policy to be adapted
                fill_prob = prob of "new" available transitions
        NOTES: Policy may need to be adapted based on goal-state convention (STAY/RESET).
        """
        T[self.A == 0] = 0  # remove impossible transitions
        T[(T == 0) & (self.A == 1)] = fill_prob  # add possible transitions
        return row_norm(T)

    def set_info_state(self, state, type=None, color=None, label=None, **kwargs):
        """Set info state on a per-state basis."""
        if type is not None:
            self.info_state.loc[state, "type"] = type
        if color is not None:
            self.info_state.loc[state, "color"] = color
            self.set_palette()
        if label is not None:
            self.info_state.loc[state, "label"] = label
        for key, value in kwargs.items():
            self.info_state.loc[state, key] = value

    def _check_statespace(self):
        """Performs various checks to ensure that state-space definition is valid."""
        if not nx.is_strongly_connected(nx.DiGraph(self.G)):
            print("ENVIRONMENT: state-space is not connected.")

    def goal_func(
        self,
        reward=10,
        stepcost=1,
        goals=3,
        goal_absorb=False,
        stay_actions=True,
        goal_remain=False,
    ):
        """
            FUNCTION: Defines reward function.
            INPUTS: reward = reward for transition to a goal state
                    stepcost = cost for transitioning to a non-goal state
                    goals = goal states (can choose several)
                    goal_absorb = absorb at goals (not yet implemented in PPP)
                    stay_actions = False implies dissipative MDP without option to stay at states
                                   True implies dissipative MDP with option to stay at states
                    goal_remain = agent forced to remain at goal
                    goal_absorb/stay_actions/goal_remain = False, default is to restart episode (wormhole from goal to start)
        """
        if not hasattr(goals, "__len__"):
            goals = [goals]
        assert np.all(np.array(goals) < self.n_state)
        self.stay_actions = stay_actions
        self.goal_absorb = goal_absorb
        self.goal_remain = goal_remain
        self.reward = reward
        self.stepcost = stepcost
        self.goals = goals

        # modify accessibility matrix self.A
        self.A_adj = self.A.copy()
        for g in self.goals:
            if self.goal_absorb:
                self.A[g, :] = 0  # no edges available from goal state (thus absorbing)
            elif self.stay_actions:
                np.fill_diagonal(
                    self.A, 1
                )  # option to stay at any state (as opposed to reset)
            elif self.goal_remain:
                self.A[g, :] = 0
                self.A[g, g] = 1
            else:
                self.A[g, :] = 0
                self.A[g, self.start] = 1  #  dissipative MDP (forced reset to start)
            self._set_state_forced()
            self.info_state.loc[g, "opt_path"] = "Goal"
            self.info_state.loc[g, "label"] = "G"
            self.info_state.loc[g, "type"] = "Goal"
        # reward function
        self.R = create_tensor((self.n_state, self.n_state), fill_value=-np.inf)
        for i in range(self.n_state):
            for j in range(self.n_state):
                if self.A[i, j] != 0:
                    if j in self.goals:
                        self.R[i, j] = reward
                    else:
                        self.R[i, j] = -stepcost
        if not self.goal_absorb and not self.stay_actions and not self.goal_remain:
            # no restart cost if goal resetting
            for goal in self.goals:
                self.R[goal, self.start] = 0
        self._compute_reward_per_state()
        self.info_state.loc[self.start, "opt_path"] = "Start"
        self.info_state.loc[self.start, "label"] = "S"
        self.info_state.loc[self.start, "type"] = "Start"
        self._label_shortest_paths()
        self._set_graph()

    def distance(self, state1, state2, interval_size=1.0):
        """distance between state1 and state2 = shortest_path_length(A) x interval_size"""
        if not hasattr(self, "shortest_n_steps"):
            self.shortest_n_steps = dict(nx.shortest_path_length(self.G))
        return self.shortest_n_steps[state1][state2] * interval_size

    def _compute_reward_per_state(self):
        """
        FUNCTION: Environment reward functions R is defined per transition.
                  This function computes a state-dependent reward function by marginalizing
                  over a random policy.
        INPUTS: self.R
        OUTPUTS: self.R_state
        NOTES: Transition reward is assigned to outcome state.
        """
        self.R_state = self.R.mean(axis=1)

    def _set_graph(self, X=None):
        """
        FUNCTION: Converts X to a networkx digraph self.G
        NOTES: A_adj is default
        """
        if X is None:
            X = self.A_adj
        self.G = nx.DiGraph(X)
        self._check_statespace()

    def stoch_mat_to_trans_weights(self):
        """converts stochastic matrix to graph edge weights and info_transition"""
        if hasattr(self, "T"):
            if hasattr(self, "G"):
                for edge in self.G.edges:
                    s1 = edge[0]
                    s2 = edge[1]
                    self.G.edges[s1, s2]["prob"] = self.T[s1, s2]
            for ix in range(self.info_transition.shape[0]):
                s = self.info_transition.loc[ix, "source"]
                t = self.info_transition.loc[ix, "target"]
                self.info_transition.loc[ix, "prob"] = self.T[s, t]

    def _set_graph_from_trans_attr(self, attr="prob"):
        edgesdf = self.info_transition
        self.G = nx.from_pandas_edgelist(
            df=edgesdf,
            source="source",
            target="target",
            edge_attr=attr,
            create_using=nx.DiGraph(),
        )
        remove_list = [
            edge for edge in self.G.edges() if self.G.edges[edge[0], edge[1]][attr] == 0
        ]
        self.G.remove_edges_from(remove_list)

    def _pos_dict(self, xymat=None):
        """Convert from xy matrix to pos_dict object used by networkx."""
        if xymat is None:
            xymat = self.xy
        pos = pos_dict(xymat)
        self.pos = pos
        return pos

    def _label_shortest_paths(self):
        """
        FUNCTION: Identifies shortest path from start to goal.
        NOTES: For multiple goals, solves for the nearest goal.
               For multiple shortest paths of equal length, records them all.
        """
        self._set_graph()
        if len(self.goals) > 1:
            # find goal with shortest path length
            current_shortest_length = np.inf
            for g in self.goals:
                goal_shortest_length = nx.shortest_path_length(
                    self.G, source=self.start, target=g
                )
                if goal_shortest_length < current_shortest_length:
                    current_shortest_length = goal_shortest_length
                    goal_shortest = g
        else:
            goal_shortest = self.goals[0]
        paths = list(
            nx.all_shortest_paths(self.G, source=self.start, target=goal_shortest)
        )
        for path in paths:
            for ix, state in enumerate(path):
                self.info_state.loc[state, "opt_path_bool"] = True
                if self.info_state.loc[state, "opt_path"] not in ["Start", "Goal"]:
                    self.info_state.loc[state, "opt_path"] = "Via state"
                self.info_state.loc[state, "opt_path_pos"] = ix

    def _info_goal(self):
        """
        Record nearest goal to a state as well as the distance to that goal.
        Adapts state color scheme to reflect task-orientation in "colors_task" scheme.
        """
        self._set_graph()
        goals = self.goals
        for state in range(self.n_state):
            shortest_lengths = nx.shortest_path_length(self.G, source=state)
            goal_shortest_lengths = np.array([shortest_lengths[i] for i in goals])
            self.info_state.loc[state, "goal_nearest"] = goals[
                goal_shortest_lengths.argmin()
            ]
            self.info_state.loc[state, "goal_dist"] = goal_shortest_lengths.min()

    def _set_state_forced(self):
        """Record states at which policy is forced"""
        self.info_state["forced"] = self.A.sum(1) <= 1

    def _state_information(self):
        """sets information about state including modules/colors etc"""
        self.info_state = pd.DataFrame(
            index=pd.Index(range(self.n_state), name="state")
        )
        self._set_state_forced()
        self.info_state["type"] = "Other"
        self.info_state["goal_nearest"] = -1
        self.info_state["goal_dist"] = np.nan
        self.info_state["opt_path"] = "Not on optimal path"
        self.info_state["opt_path_pos"] = np.nan
        self.info_state["opt_path_bool"] = False
        self.info_state["label"] = (self.info_state.index + 1).astype(
            "str"
        )  #  label states starting at 1
        self.info_state["x"] = np.nan
        self.info_state["y"] = np.nan
        self.info_state = self.info_state.astype(
            dtype={
                "forced": "bool",
                "type": "str",
                "goal_nearest": "int",
                "goal_dist": "float",
                "opt_path": "str",
                "opt_path_pos": "float",
                "opt_path_bool": "bool",
                "x": "float",
                "y": "float",
            }
        )

    def _transition_information(self):
        """sets information about transitions including e.g. weights"""
        if hasattr(self, "G"):
            self.info_transition = nx.to_pandas_edgelist(self.G)
        elif hasattr(self, "W"):
            self.G = nx.DiGraph(self.W)
            self.info_transition = nx.to_pandas_edgelist(self.G)
        elif hasattr(self, "A"):
            self.G = nx.DiGraph(self.A)
            self.info_transition = nx.to_pandas_edgelist(self.G)
        else:
            raise ValueError("Need source for transition information")

    def _define_state_coordinates(self):
        """copies self.xy generated by self._node_info to self.info_state['x','y']"""
        assert hasattr(self, "xy"), "xy node positions unavailable"
        self.info_state["x"] = self.xy[:, 0]
        self.info_state["y"] = self.xy[:, 1]

    def _retrieve_state_coordinates(self, state):
        """returns graph ambient space coordinates, state can be a state int of list/array of states"""
        if hasattr(state, "__len__"):
            return (
                self.info_state.loc[state, ["x", "y"]]
                .values.reshape((len(state), 2))
                .squeeze()
            )
        else:
            return self.info_state.loc[state, ["x", "y"]].values.flatten()

    def set_palette(self, var=None):
        """Copies info_state.color information to palette grouped by var."""
        if var is None:
            var = "hue"
        self.viz_kwargs["palette"] = (
            self.info_state.groupby(self.viz_kwargs[var])
            .apply(lambda x: x.loc[x.index[0], "color"])
            .to_dict()
        )
        self.viz_kwargs_lines["palette"] = self.viz_kwargs["palette"]
        self.viz_kwargs_markers["palette"] = self.viz_kwargs["palette"]

    def set_viz_scheme(
        self, alphas=[0.3, 1.0], sizes_lines=(3, 1), sizes_markers=(300, 200)
    ):
        """
        FUNCTION: Sets plotting variables according to viz_scheme and info_state
        INPUTS: alphas = transparency values for minor and major plot components respectively
                sizes_lines = range of line sizes (inverted for "goal_dist" by default)
                sizes_markers = range of marker sizes (inverted for "goal_dist" by default)
        """
        self.viz_scheme = "default"
        size_lines = "goal_dist"
        size_markers = "KL_prior"
        dashes = {
            "Bottleneck": (2, 2, 10, 2),
            "Start": "",
            "Goal": "",
            "Switch": "",
            "Via": "",
            "Other": "",
        }
        markers = {
            "Bottleneck": "P",
            "Start": "o",
            "Goal": "o",
            "Switch": "o",
            "Via": "o",
            "Other": ".",
        }
        style_order = ["Bottleneck", "Start", "Goal", "Switch", "Via", "Other"]
        hue = "state"
        hue_order = None

        self.viz_kwargs = {
            "legend": None,
            "units": "state",
            "hue": hue,
            "hue_order": hue_order,
            "style": "type",
            "style_order": style_order,
            "estimator": None,
        }
        self.viz_kwargs_markers = {
            **self.viz_kwargs,
            "size": size_markers,
            "sizes": sizes_markers,
            "markers": markers,
        }
        self.viz_kwargs_lines = {
            **self.viz_kwargs,
            "size": size_lines,
            "sizes": sizes_lines,
            "dashes": dashes,
        }
        self.state_type_major = [
            "Bottleneck",
            "Start",
            "Goal",
            "Via",
            "Switch",
        ]
        self.state_type_alphas = {
            "minor": alphas[0],
            "major": alphas[1],
        }

        # construct palettes
        if self.viz_scheme == "default":
            self.color_palette = ["grey"]
            self.info_state.loc[:, "color_index"] = 0
            self.info_state["color"] = self.info_state.color_index.apply(
                lambda x: self.color_palette[int(x)]
            )
        self.info_state = self.info_state.astype(
            dtype={"color_index": "int", "color": "str"}
        )
        self.set_palette()
        if hasattr(self, "goals"):
            self._info_goal()

    def plot_graph(self, width=2000, height=2000, dpi=300):
        """
        FUNCTION: Saves a state-space graph plot.
        """
        # color settings
        node_color = "color"

        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition

        # other settings
        if self.__type__ in ["decision-tree"]:
            directed = True
        else:
            directed = False

        nodes = hv.Nodes(
            data=nodesdf, kdims=["x", "y", "state"], vdims=["type", "color", "label"]
        )
        graph = hv.Graph(
            data=(edgesdf, nodes),
            kdims=["source", "target"],
            vdims=["weight"],
            label=self.__name__,
        )
        labels = hv.Labels(nodes, ["x", "y"], "label").opts(
            text_font_size=text_font_size,
            text_color="black",
            show_frame=False,
            toolbar="disable",
        )

        graph.opts(
            title="",
            directed=directed,
            padding=0.1,
            bgcolor="white",
            width=width,
            height=height,
            show_frame=False,
            xaxis=None,
            yaxis=None,
            node_size=node_size,
            node_color=node_color,
            edge_line_width=edge_size,
            edge_color_index=color_index_edge,
            edge_cmap=cmap_edge,
            edge_line_color="black",
            toolbar="disable",
        )
        graph = graph * labels
        hv.save(graph, filename=self.fname_graph, backend="bokeh", dpi=dpi)

    def plot_stochastic_matrix(self):
        """Plots stochastic matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.T, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("stochastic matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_policy_matrix(self):
        """Plots stochastic matrix"""
        assert hasattr(self, "PI"), "no policy set"
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.PI, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("policy matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_access_matrix(self):
        """Plots accessibility matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.A, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.title("access matrix")
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_adjacency_matrix(self):
        """Plots accessibility matrix"""
        plt.figure(figsize=(12, 10))
        im = plt.imshow(self.A_adj, origin="upper", cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.colorbar(im, shrink=1)
        plt.xlabel("future state")
        plt.ylabel("current state")
        plt.gca().grid(color="gray", linestyle="-", linewidth=0.5)

    def plot_environment(self, X=None, ax=None, figsize=(12, 12)):
        """
        FUNCTION: plot environment graph
        """
        if X is not None:
            self._set_graph(X=X)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.axes(ax)
        self.draw_graph(ax=ax)

    def draw_graph(self, with_labels=False, ax=None):
        """uses networkx to draw graph"""
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        pos = self._pos_dict()
        edge_weights = nx.get_edge_attributes(self.G, "prob").values()
        edge_weights = [10 * e for e in edge_weights]
        nx.draw(
            self.G,
            pos,
            node_size=30,
            alpha=0.7,
            node_color="black",
            width=0,
            with_labels=with_labels,
            font_size=24,
            ax=ax,
            arrows=False,
        )
        nx.draw_networkx_edges(
            G=self.G, pos=pos, width=edge_weights, alpha=0.6, arrows=False, ax=ax
        )
        plt.axis("equal")
        plt.axis("off")

    def plot_state_func(
        self,
        state_vals,
        vlims=[None, None],
        ax=None,
        annotate=False,
        cmap=plt.cm.autumn,
        cbar=False,
        cbar_label="",
        node_edge_color="black",
        **kwargs
    ):
        """
        FUNCTION: plots state function state_vals on world_array imshow.
        INPUTS: state_vals = state values of function to plot
                vlims = [vmin,vmax] value range
                ax = figure axis to plot to
                annotate = textualize function values on states
                cmap = colormap
                cbar = include offset colorbar
                cbar_label = label for colorbar
        """
        if ax is None:
            ax = self.plot_environment(ax=ax)
        state_vals_dict = {}
        for ix, state in enumerate(self.info_state.index):
            state_vals_dict[state] = state_vals[ix]
        nx.set_node_attributes(G=self.G, name="state_val", values=state_vals_dict)
        node_colors = [self.G.nodes[n]["state_val"] for n in self.G.nodes]
        ec = nx.draw_networkx_edges(G=self.G, pos=self.pos, ax=ax, **kwargs)
        nc = nx.draw_networkx_nodes(
            G=self.G,
            pos=self.pos,
            node_size=300,
            node_color=node_colors,
            cmap=cmap,
            ax=ax,
            **kwargs
        )
        if node_edge_color is not None:
            nc.set_edgecolor(node_edge_color)
            nc.set_linewidth(1)
        # create text annotations
        if annotate:
            for state in range(self.n_state):
                x = self.xy[state, 0]
                y = self.xy[state, 1]
                state_val = state_vals[state]
                if not np.isnan(state_val):
                    text = ax.text(x, y, state_val, ha="center", va="center", color="k")
        remove_axes(ax)
        ax.axis("equal")
        if cbar:
            fig = plt.gcf()
            cbar = fig.colorbar(nc, shrink=0.6, orientation="horizontal", pad=0)
            if cbar_label != "":
                cbar.set_label(cbar_label)
        return ax