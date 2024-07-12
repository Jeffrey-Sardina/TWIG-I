class Early_Stopper:
    def __init__(self, start_epoch, patience, mode, precision):
        '''
        __init__() creates the Early_Stopper object. This class serves to provide an simple way to define an early-stopping protocol for TWIG-I.

        NOTE that epochs are 1-indexed, so the first epoch is epoch 1.

        The arguments it accepts are:
            - start_epoch (int): the first epoch on which the early stopping could be applied (this is often called a `burn-in` time)
            - patience (int): the number of epochs to wait after seeing what would otherwise be a stop-condition before the early stopping is actually applied.
            - mode (str): the mode of early stopping to use. Options are as follows:
                - "on-falter" -- trigger early stopping the first time a validation result does not get better than a previous result
                - "never" -- never do early stopping
            - precision (int): the number of decimal points to consider when testing to change in MRR.

        The values it returns are:
            - None
        '''
        self.start_epoch = start_epoch
        self.patience = patience
        self.mode = mode
        self.precision = precision
        self.validation_results = []
        self.valid_modes = ("on-falter", "never")


        # input validation
        assert type(start_epoch) == int and start_epoch >= 0
        assert type(patience) == int
        assert type(precision) == int and precision >= 1
        assert self.mode in self.valid_modes, f"Unknown mode: {self.mode}. Mode must be one of: {self.valid_modes}"


    def assess_validation_result(self, epoch_num, valid_mrr):
        '''
        assess_validation_result() examines the current validation result and returns a bool that thells TWIG whether it should trigger early stopping or not.

        The arguments is accepts are:
            - epoch_num (int): the current epoch
            - valid_mrr (float): the MRR valua acheived on the validation round for the given (current) epoch

        The values it returns are:
            - should_stop (bool): True if TWIG-I should trigger early stopping, False otherwise
        '''
        should_stop = False
        if self.mode == "never":
            return should_stop
        
        prev_epoch, prev_mrr_int = self.validation_results[-1]
        valid_mrr_int = int(round((valid_mrr * 10  ** self.precision), 0)
        self.validation_results.append((epoch_num, valid_mrr_int))

        if self.mode == "on-falter":
            if prev_mrr_int >= valid_mrr_int and \
                epoch_num - prev_epoch >= self.patience and \
                epoch_num > self.start_epoch:
                should_stop = True
        else:
            assert False, f"Unknown mode: {self.mode}. Mode must be one of: {self.valid_modes}"
        return should_stop
