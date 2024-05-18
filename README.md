(root)
  │
  ├── images                         <- The validate image
  ├── labels                         <- The validate label
  │
  ├── runs
  │   ├── detect
  │   |   └── predict                <- Containing the validated image
  │   └── nlvnpf-0137-01-045.txt     
  │   └── nlvnpf-0137-01-046.txt     <- Validated label
  │   └──...
  ├── best1.pt                       <- Trained Model
  │
  ├── excute.py                      <- Validating character
  │
  ├── label.py                       <- Validating label


Run `excute.py` to generate validated images
Run `label.py` to generate validated labels


