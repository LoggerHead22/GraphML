# GraphML

## Описание репозитория
 
  - Papers - собрание статей на тему Graph Neural Network
  
  - Result - таблицы с результатами экспериментов 
  
  - Main - cкрипты для обучения GNN

  - dataset - набор публичных датасетов в формате smiles. Включает в себя как регресссионные, так и классификационные задачи. Таргеты содержатся в файлах targets.csv.

## Пример запуска обучения

	python gnn/train.py --data_path dataset/FreeSolv/train_w_targets.csv --separate_test_path dataset/FreeSolv/test_w_targets.csv --dataset_type regression --save_dir checkpoints/FreeSolv_checkpoint --metric rmse --extra_metric r2
	
	