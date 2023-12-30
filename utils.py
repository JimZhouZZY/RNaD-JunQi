from config import _NUM_ACTORS, _BATCH_SIZE, _SAVE_GAP, _FILE_PATH
def print_loss(loss, learner_steps, i):
    if i % 1 == 0:
        print(
            f"[DATA] Loss: {round(loss, 4)}, "
              f"Actor Steps: {learner_steps*_BATCH_SIZE}, "
              f"Learner Steps: {learner_steps}"
        )
