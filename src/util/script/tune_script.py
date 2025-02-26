import src

def TuneScript(model_version, data_version):

    match model_version:

        case 'alpha':

            model = src.model.load(model_version)
            data = src.data.load(data_version)

            data_validation_pipeline = src.core.pipeline.DataValidationPipeline({'version': 'alpha'})
            data_clean = data_validation_pipeline.process(data['data'])

            engine = src.core.Engine()
            ancestor = engine.produce('alpha')
            ancestor.load_state_dict(model['model'])

            model_embedding_pipeline = src.core.pipeline.ModelEmbeddingPipeline({'version': 'alpha'})
            data_loader = model_embedding_pipeline.process(data_clean)

            model_training_pipeline = src.core.pipeline.ModelTrainingPipeline(ancestor, data_loader, {'version': 'alpha'})
            model_training_pipeline.train()

            config = {'version': 'alpha', 'message_length': model_embedding_pipeline.message_length}
            environment = {'encoder' : model_embedding_pipeline.encoder}

            src.model.save(model_training_pipeline.model, environment, config)
