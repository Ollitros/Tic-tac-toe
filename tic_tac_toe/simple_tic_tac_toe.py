import numpy as np


class TicTacToe:

    def __init__(self, game_option=1):

        assert game_option == 1 or game_option == 2

        self.game_option = game_option
        self.actions = None
        self.states = None

    def reset(self):

        self.actions = np.zeros((4))
        self.states = np.zeros((4))

        return self.states

    def step(self, action, side=1):

        new_s, r, done = None, None, None

        a11 = 0
        a22 = 3

        a12 = 1
        a21 = 2

        if self.game_option == 1:
            # # Again in the same point
            # if self.states[action] == 1:
            #
            #     r = -5
            #     done = False
            #     self.states[action] = 1
            #     new_s = self.states

            # In the new point
            if self.states[action] == 0:

                self.states[action] = 1

                # Victory
                if self.states[a11] == self.states[a22] == 1:

                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states

                # Victory
                elif self.states[a12] == self.states[a21] == 1:

                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states

                # Casual step
                else:

                    r = -1
                    done = False
                    self.states[action] = 1
                    new_s = self.states
        else:

            # In the new point
            if self.states[action] == 0:

                self.states[action] = side

                # Victory
                if self.states[a11] == self.states[a22] == side:

                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states

                # Victory
                elif self.states[a12] == self.states[a21] == side:

                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states

                # Casual step
                else:

                    r = -1
                    done = False
                    self.states[action] = side
                    new_s = self.states

        return new_s, r, done

    @property
    def n(self):
        return self.actions.size
