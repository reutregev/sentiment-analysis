import logging


class Logger:
    def __init__(self):
        # Initiating the logger object
        self.logger = logging.getLogger(__name__)

        # Set the level of the logger
        self.logger.setLevel(logging.DEBUG)

        # Create the logs.log file with a specified logs structure
        file_handler = logging.FileHandler('../logs.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Print the logs to the console as well
        console_andler = logging.StreamHandler()
        console_andler.setFormatter(formatter)
        self.logger.addHandler(console_andler)


logger = Logger().logger
