import numpy as np


class TicTacToe:

    def __init__(self, game_mode=1):

        assert game_mode == 1 or game_mode == 2

        self.game_mode = game_mode
        self.actions = None
        self.states = None

    def reset(self):

        self.actions = np.zeros((9))
        self.states = np.zeros((9))

        return self.states

    def step(self, action, side=1):

        new_s, r, done = None, None, None

        a11, a12, a13 = 0, 1, 2
        a21, a22, a23 = 3, 4, 5
        a31, a32, a33 = 6, 7, 8

        if self.game_mode == 1:
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
                # Victory 1
                if self.states[a11] == self.states[a12] == self.states[a13] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 2
                elif self.states[a21] == self.states[a22] == self.states[a23] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 3
                elif self.states[a31] == self.states[a32] == self.states[a33] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 4
                elif self.states[a11] == self.states[a21] == self.states[a31] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 5
                elif self.states[a12] == self.states[a22] == self.states[a32] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 6
                elif self.states[a13] == self.states[a23] == self.states[a33] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 7
                elif self.states[a11] == self.states[a22] == self.states[a33] == 1:
                    r = 10
                    done = True
                    self.states[action] = 1
                    new_s = self.states
                # Victory 8
                elif self.states[a13] == self.states[a22] == self.states[a31] == 1:
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

            if self.states[action] == 0:
                self.states[action] = side
                # Victory 1
                if self.states[a11] == self.states[a12] == self.states[a13] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 2
                elif self.states[a21] == self.states[a22] == self.states[a23] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 3
                elif self.states[a31] == self.states[a32] == self.states[a33] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 4
                elif self.states[a11] == self.states[a21] == self.states[a31] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 5
                elif self.states[a12] == self.states[a22] == self.states[a32] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 6
                elif self.states[a13] == self.states[a23] == self.states[a33] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 7
                elif self.states[a11] == self.states[a22] == self.states[a33] == side:
                    r = 10
                    done = True
                    self.states[action] = side
                    new_s = self.states
                # Victory 8
                elif self.states[a13] == self.states[a22] == self.states[a31] == side:
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
