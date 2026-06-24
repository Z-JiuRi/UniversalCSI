adapter=mlp gpu=1 seed=2026 bash scripts/train_adapter.sh
adapter=mlp_direct gpu=1 seed=2026 bash scripts/train_adapter.sh

adapter=mlp gpu=1 seed=1024 bash scripts/train_adapter.sh
adapter=mlp_direct gpu=1 seed=1024 bash scripts/train_adapter.sh

adapter=mlp gpu=4 seed=0 bash scripts/train_adapter.sh
adapter=mlp_direct gpu=4 seed=0 bash scripts/train_adapter.sh

adapter=mlp gpu=4 seed=520 bash scripts/train_adapter.sh
adapter=mlp_direct gpu=4 seed=520 bash scripts/train_adapter.sh

adapter=mlp gpu=4 seed=796 bash scripts/train_adapter.sh
adapter=mlp_direct gpu=4 seed=796 bash scripts/train_adapter.sh

