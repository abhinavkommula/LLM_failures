from time import gmtime, strftime
import os

class Logger:
    def __init__(self):
        self.cur_file = ""
                
    def change_job(self, task_name):
        #if task_name != "" and os.path.isfile("metrics/logs/" + task_name + ".txt"):
        #    os.remove("metrics/logs/" + task_name + ".txt")

        self.cur_file = "metrics/logs/" + task_name + ".txt"

    def print(self, text):
        if self.cur_file == "":
            return

        with open(self.cur_file, "a") as f:
            f.write(f"|INFO|{strftime('%Y-%m-%d %H:%M:%S', gmtime())}| {text}\n")

    def finish(self):
        if self.cur_file == "":
            return

        with open(self.cur_file, "a") as f:
            f.write(f"|FINISH|{strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n")

    def check_finish(self, task_name):
        path = "metrics/logs/" + task_name + ".txt"
        
        if not os.path.isfile(path):
            return False

        with open(path, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)

                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)

            except OSError:
                f.seek(0)		

            last_line = f.readline().decode()

            if "|FINISH|" in last_line:
                return True

        return False

