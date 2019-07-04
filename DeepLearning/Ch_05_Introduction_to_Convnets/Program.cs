using System;
using DeepLearningWithCNTK;
using feed_t = System.Collections.Generic.Dictionary<CNTK.Variable, CNTK.Value>;
using test_feed_t = CNTK.UnorderedMapVariableValuePtr;

namespace Ch_05_Introduction_to_Convnets
{
    class Program
    {
        CNTK.Function m_network;
        CNTK.Function m_lossFunction;
        CNTK.Function m_accuracyFunction;
        CNTK.Trainer m_trainer;
        CNTK.Evaluator m_evaluator;

        CNTK.Variable m_imageTensor;
        CNTK.Variable m_labelTensor;
        CNTK.DeviceDescriptor m_computeDevice;

        float[][] m_trainImages;
        float[][] m_testImages;
        float[][] m_trainLabels;
        float[][] m_testLabels;

        static void Main()
        {
            new Program().Run();
        }

        void LoadData()
        {
            if (!System.IO.File.Exists("train_images.bin"))
            {
                System.IO.Compression.ZipFile.ExtractToDirectory("mnist_data.zip", ".");
            }
            m_trainImages = Util.load_binary_file("train_images.bin", 60000, 28 * 28);
            m_testImages = Util.load_binary_file("test_images.bin", 10000, 28 * 28);
            m_trainLabels = Util.load_binary_file("train_labels.bin", 60000, 10);
            m_testLabels = Util.load_binary_file("test_labels.bin", 60000, 10);
            Console.WriteLine("Done with loading data\n");
        }


        void CreateNetwork()
        {
            m_computeDevice = Util.get_compute_device();
            Console.WriteLine("Compute Device: " + m_computeDevice.AsString());

            m_imageTensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 28, 28, 1 }), CNTK.DataType.Float);
            m_labelTensor = CNTK.Variable.InputVariable(CNTK.NDShape.CreateNDShape(new int[] { 10 }), CNTK.DataType.Float);

            m_network = m_imageTensor;
            m_network = Util.Convolution2DWithReLU(m_network, 32, new int[] { 3, 3 }, m_computeDevice);
            m_network = CNTK.CNTKLib.Pooling(m_network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
            m_network = Util.Convolution2DWithReLU(m_network, 64, new int[] { 3, 3 }, m_computeDevice);
            m_network = CNTK.CNTKLib.Pooling(m_network, CNTK.PoolingType.Max, new int[] { 2, 2 }, new int[] { 2 });
            m_network = Util.Convolution2DWithReLU(m_network, 64, new int[] { 3, 3 }, m_computeDevice);
            m_network = Util.Dense(m_network, 64, m_computeDevice);
            m_network = CNTK.CNTKLib.ReLU(m_network);
            m_network = Util.Dense(m_network, 10, m_computeDevice);

            Util.log_number_of_parameters(m_network);

            m_lossFunction = CNTK.CNTKLib.CrossEntropyWithSoftmax(m_network.Output, m_labelTensor);
            m_accuracyFunction = CNTK.CNTKLib.ClassificationError(m_network.Output, m_labelTensor);

            var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)m_network.Parameters());
            var learner = CNTK.CNTKLib.AdamLearner(parameterVector, new CNTK.TrainingParameterScheduleDouble(0.001, 1), new CNTK.TrainingParameterScheduleDouble(0.99, 1));
            m_trainer = CNTK.CNTKLib.CreateTrainer(m_network, m_lossFunction, m_accuracyFunction, new CNTK.LearnerVector() { learner });
            m_evaluator = CNTK.CNTKLib.CreateEvaluator(m_accuracyFunction);

        }

        void TrainNetwork()
        {
            int epochs = 5;
            int batch_size = 64;

            for (int current_epoch = 0; current_epoch < epochs; current_epoch++)
            {

                // training phase        
                var train_indices = Util.shuffled_indices(60000);
                var pos = 0;
                while (pos < train_indices.Length)
                {
                    var pos_end = Math.Min(pos + batch_size, train_indices.Length);
                    var minibatch_images = Util.get_tensors(m_imageTensor.Shape, m_trainImages, train_indices, pos, pos_end, m_computeDevice);
                    var minibatch_labels = Util.get_tensors(m_labelTensor.Shape, m_trainLabels, train_indices, pos, pos_end, m_computeDevice);
                    var feed_dictionary = new feed_t() { { m_imageTensor, minibatch_images }, { m_labelTensor, minibatch_labels } };
                    m_trainer.TrainMinibatch(feed_dictionary, false, m_computeDevice);
                    pos = pos_end;
                }

                // evaluation phase
                var accuracy = 0.0;
                var num_batches = 0;
                pos = 0;
                while (pos < m_testImages.Length)
                {
                    var pos_end = Math.Min(pos + batch_size, m_testImages.Length);
                    var minibatch_images = Util.get_tensors(m_imageTensor.Shape, m_testImages, pos, pos_end, m_computeDevice);
                    var minibatch_labels = Util.get_tensors(m_labelTensor.Shape, m_testLabels, pos, pos_end, m_computeDevice);
                    var feed_dictionary = new test_feed_t() { { m_imageTensor, minibatch_images }, { m_labelTensor, minibatch_labels } };
                    var minibatch_accuracy = m_evaluator.TestMinibatch(feed_dictionary, m_computeDevice);
                    accuracy += minibatch_accuracy;
                    pos = pos_end;
                    num_batches++;
                }
                accuracy /= num_batches;
                Console.WriteLine(string.Format("Epoch {0}/{1}, Test accuracy:{2:F3}", current_epoch + 1, epochs, 1.0 - accuracy));
            }
        }

        void Run()
        {
            LoadData();
            CreateNetwork();
            TrainNetwork();
        }
    }
}
