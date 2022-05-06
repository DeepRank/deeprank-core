class UnknownAtomError(ValueError):
    "should be raised when an unknown atom type is encountered"
    # Constructor method
    def __init__(self, value):
        super().__init__(self)
        self.value = value
    # __str__ display function
    def __str__(self):
        return(repr(self.value))