class Config:
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    validating_dir = "./data/faces/testing/"
    webcam_dir = "./data/faces/webcam/"
    save_path = "./models/model.pt"
    train_batch_size = 64  # original = 64
    valid_batch_size = 16
    learning_rate = 0.0015  # original = 0.0005
    dynamic_lr = 5
    save_rate = 5
    transform_rate = 1
    train_number_epochs = 1001
    threshold = 0.95
    model_type = "threshold"  # threshold, absolute, resume_training, resnet, resnet_cat
    run_type = "train"  # train, test, webcam
    loss_type = "contrastive"  # contrastive, cross_entropy
    max_rotation = 0  # 20
    p_hor_flip = 0
    p_ver_flip = 0
    weight_decay = 1e-5
    MNIST = False
    LFW = False
    training_size = None
    number_of_tests = 1
    optimizer = "adam"  # adam, sgd
    full_test = True
    full_test_threshold_start = 0.5
    full_test_threshold_end = 2.5
    full_test_threshold_step = 0.01
