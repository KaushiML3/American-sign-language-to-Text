
from src.preprocess import preprocess
from src.training import model_build
from src.common.custom_logger import setup_logger

logger=setup_logger("trainer")

if __name__ == "__main__":

    preprocess_step=not None
    if preprocess_step is not None:
        attributes = {"uri" :"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "database":'asl_land_mark_detection',
            "collection":'original',
            "IQR_collection":"IQR_collection2",
            "ISO_collection":"ISO_collection2"}

        pre_pro=preprocess(attributes,file_path=None)
        df=pre_pro.load_dataset()
        pre_pro.main_exploratory(df)

    else:
        pass


    attributes2 = {"uri" :"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "database":'asl_land_mark_detection',
            "collection":'ISO_collection'}

    build=model_build(attributes=attributes2,file_path=None)
    logger.info("############### Training rf model ############################")
    build.build_rf(save_path="rf_model",n_estimators=200)

    logger.info("############### Training BGC model ############################")
    build.build_bg(save_path="bgc_model",n_estimators=10,max_depth=8)

    logger.info("############### Training VBC model ############################")
    build.build_CatBC(save_path="cbc_model",iterations=20,max_depth=12)


    logger.info("############### Training XBC model ############################")
    params = {
    'objective': 'binary:logistic',  # for binary classification
    'max_depth': 5,
    'eta': 0.1,  # learning rate
    'eval_metric': 'logloss'
}
    build.build_xbc(save_path="xbc_model",params=params)