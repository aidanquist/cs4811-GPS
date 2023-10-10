"""
    Include libraries:
    heapq: Heap queue is a special tree structure in which each parent node is less than or equal to its child node.
    This library is used to implement priority queues where the queue item with a higher weight is given more priority in processing.
"""

import heapq
import copy


class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class GPSSolver:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def heuristic(self, state):
        pass

    def actions(self, state):
        pass

    def result(self, state, action):
        pass

    def solve(self):
        """
            Implementation of A* Search to find a solution to the problem
            defined by the initial state and goal state
        """

        # Initialize the frontier as a priority queue and the explored set.
        frontier = []
        explored = set()

        # Create a Node object for the initial state and add it to the frontier.
        initial_node = Node(self.initial_state)
        heapq.heappush(frontier, initial_node)

        # Main loop for A* search.
        while frontier:
            # Pop the node with the lowest total cost (including cost and heuristic) from the frontier.
            current_node = heapq.heappop(frontier)
            current_state = current_node.state

            # Check if the current state is the goal state.
            if current_state == self.goal_state:
                # If yes, return the path to reach the goal state.
                return self.construct_path(current_node)

            # Convert the current state to a tuple before adding it to the explored set.
            explored.add(tuple(map(tuple, current_state)))

            # Explore possible actions from the current state.
            for action in self.actions(current_state):
                # Generate the next state based on the action.
                next_state = self.result(current_state, action)

                # Convert the next state to a tuple to check if it's in the explored set.
                next_state_tuple = tuple(map(tuple, next_state))

                # Check if the next state has not been explored yet.
                if next_state_tuple not in explored:
                    # Calculate the cost to reach the next state from the current state.
                    cost = current_node.cost + 1

                    # Calculate the heuristic estimate for the next state.
                    heuristic = self.heuristic(next_state)

                    # Create a new node representing the next state and add it to the frontier for further exploration.
                    child_node = Node(next_state, current_node, action, cost, heuristic)
                    heapq.heappush(frontier, child_node)

        # If no solution is found, return None to indicate that there is no solution.
        return None

    def construct_path(self, node):
        path = []
        while node:
            if node.action:
                path.append(node.action)
            node = node.parent
        path.reverse()
        return path


class EightPuzzleGPS(GPSSolver):
    def heuristic(self, state):
        """
            - Calculates the heuristic estimate for a given state.
            - Computes the number of misplaced tiles in the current state compared to the goal state.
            - This heuristic is admissible because it never overestimates the cost of reaching the goal.
        """
        misplaced = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != self.goal_state[i][j]:
                    misplaced += 1
        return misplaced

    def actions(self, state):
        """
            - Valid actions depend on the position of the empty tile (represented as 0 in the state).
            - Can move tiles Up, Down, Left, or Right, as long as the move is
              within the bounds of the puzzle.
            - Returns a list of valid actions (moves) that can be taken from the current state.
        """

        actions = []
        empty_i, empty_j = self.find_empty(state)
        if empty_i > 0:
            actions.append("Up")
        if empty_i < 2:
            actions.append("Down")
        if empty_j > 0:
            actions.append("Left")
        if empty_j < 2:
            actions.append("Right")
        return actions

    def result(self, state, action):
        """
         - Applies a given action to the current state to produce the resulting state.
         - It simulates the movement of the empty tile (0) and the adjacent tile in the
         specified direction (Up, Down, Left, or Right).
         - Returns a new state that represents the puzzle after the action is taken.
        """

        empty_i, empty_j = self.find_empty(state)
        new_state = [list(row) for row in state]
        if action == "Up":
            new_state[empty_i][empty_j], new_state[empty_i - 1][empty_j] = new_state[empty_i - 1][empty_j], \
                new_state[empty_i][empty_j]
        elif action == "Down":
            new_state[empty_i][empty_j], new_state[empty_i + 1][empty_j] = new_state[empty_i + 1][empty_j], \
                new_state[empty_i][empty_j]
        elif action == "Left":
            new_state[empty_i][empty_j], new_state[empty_i][empty_j - 1] = new_state[empty_i][empty_j - 1], \
                new_state[empty_i][empty_j]
        elif action == "Right":
            new_state[empty_i][empty_j], new_state[empty_i][empty_j + 1] = new_state[empty_i][empty_j + 1], \
                new_state[empty_i][empty_j]
        return tuple(tuple(row) for row in new_state)

    def find_empty(self, state):
        """
         - A helper method that finds the position of the empty tile (0) in the current state.
         - Returns the row and column indices of the empty tile.
        """

        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j


class WorldBlockGPS(GPSSolver):
    def heuristic(self, state):
        """
        - Calculates the heuristic estimate for a given state in the World Block problem.
        - The heuristic is defined as the number of blocks that are not in their correct
            positions compared to the goal state.
        - Helps to guide the A* search algorithm towards finding a solution efficiently.
        """

        # YOUR CODE GOES HERE (A)

        misplaced = 0
        if len(state) == len(goal_state_world_block):
            # checks to make sure row lengths match
            for i in range(len(state)):
                if len(state[i]) == len(goal_state_world_block[i]):
                    # checks to make sure each column length matches
                    for j in range(len(state[i])):
                        if state[i][j] != goal_state_world_block[i][j]:
                            misplaced += 1
        return misplaced
        # returns number of spaces that don't match
        # YOUR CODE ENDS HERE

    def actions(self, state):
        """
       - In the World Block problem, a valid action consists of moving a block from
           one stack (source_stack) to another stack (target_stack).
       - The code loops through the stacks in the current state and generates all
           possible block-moving actions.
       - Returns a list of valid actions (moves) that can be taken from the current state.

        input: a list of lists (states)
               Example: [['B'], [], ['A']]
        output: a list of movements
                eg: [(0, 1), (0, 2), (2, 0), (2, 1)]
       """

        # Define valid actions for the World Block problem.
        actions = []




        # YOUR CODE GOES HERE (B)

        # YOUR CODE ENDS HERE

        return actions

    def result(self, state, action):
        """
        Description:
        This method applies a given action to the current state to produce the resulting state.
        It simulates the movement of a block from the source_stack to the target_stack and

        Returns a new state that represents the problem after the action is taken.

        Input: state (a list of lists) and actions (list of movement)
        output: A list of list, new state

        Hint: Observe the flow in other example given.
        Observe the possible moves and generate new state when given input is applied against previous state.
        """

        # Apply the action to the state to generate the resulting state.
        source_stack, target_stack = action
        new_state = copy.deepcopy(state)

        # YOUR CODE GOES HERE (C)

        # YOUR CODE ENDS HERE

        return new_state


if __name__ == "__main__":
    # 8-puzzle example
    initial_state_8puzzle = ((7, 2, 4), (5, 0, 6), (8, 3, 1))
    goal_state_8puzzle = ((0, 1, 2), (3, 4, 5), (6, 7, 8))

    gps_8puzzle = EightPuzzleGPS(copy.deepcopy(initial_state_8puzzle), goal_state_8puzzle)
    solution_8puzzle = gps_8puzzle.solve()

    if solution_8puzzle:
        print("8-Puzzle Solution found:")
        for action in solution_8puzzle:
            print(action)
    else:
        print("8-Puzzle: No solution found.")

    # World Block example (You need to provide initial_state and goal_state)
    initial_state_world_block = None  # Provide the initial state for the World Block problem
    goal_state_world_block = None  # Provide the goal state for the World Block problem

    initial_state_world_block = [
        ["B"],  # Row 1: Block A and Block B are initially stacked in the first column
        [],
        ["A"],
    ]

    goal_state_world_block = [
        ["A"],  # Goal state: Block A and Block B should be stacked on second row.
        [],
        ["B"],
    ]

    if initial_state_world_block and goal_state_world_block:
        gps_world_block = WorldBlockGPS(copy.deepcopy(initial_state_world_block), goal_state_world_block)
        solution_world_block = gps_world_block.solve()

        if solution_world_block:
            print("\nWorld Block Solution found:")
            for action in solution_world_block:
                print(action)
        else:
            print("World Block: No solution found.")
    else:
        print("\nWorld Block: Please provide initial and goal states.")
