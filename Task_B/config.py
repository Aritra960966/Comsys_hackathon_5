"""
Configuration module for TaskB Face Recognition
"""

CONFIG = {
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 3e-4,
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 8,
    'MARGIN': 1.2,
    'ALPHA': 0.2,
    'GAMMA': 2,
    'WARMUP_EPOCHS': 5,
    'TRAIN_DIR': '/content/dataset/Comys_Hackathon5/Task_B/train',
    'VAL_DIR': '/content/dataset/Comys_Hackathon5/Task_B/val',
    'TEST_DIR': '/content/dataset/Comys_Hackathon5/Task_B/test',
    'MODEL_PATH': 'taskB_best_model.pth',
    'SIMILARITY_MODEL_PATH': 'taskB_best_similarity_model.pth',
    'DISTANCE_MODEL_PATH': 'taskB_best_distance_model.pth',
    'OPTIMAL_SIM_MODEL_PATH': 'taskB_optimal_similarity_model.pth',
    'OPTIMAL_DIST_MODEL_PATH': 'taskB_optimal_distance_model.pth',
    'CSV_LOG_PATH': 'taskB_metrics_log.csv',
    'THRESHOLD': 0.5,
    'THRESHOLD_SEARCH_RANGE': (0.1, 0.9),
    'THRESHOLD_SEARCH_STEPS': 20,
    'EMBEDDING_DIM': 256,
    'DROPOUT_RATE': 0.3,
    'USE_MIXUP': False,
    'MIXUP_ALPHA': 0.4
}