using System;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Net.Http;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using TensorFlow;
using System.Drawing;
using System.Collections.Generic;

namespace Coach {
    public struct ImageDims {
        public int InputSize { get; set; }
        public int ImageMean { get; set; }
        public float ImageStd { get; set; }

        public ImageDims(int inputSize, int imageMean, float imageStd) {
            this.InputSize = inputSize;
            this.ImageMean = imageMean;
            this.ImageStd = imageStd;
        }
    }

    public static class ImageUtil
    {
        public static Bitmap BitmapFromFile(string filePath) {
            return new Bitmap(filePath, true);
        }

        public static Bitmap BitmapFromBytes(byte[] input) {
            Bitmap bmp;
            using (var ms = new MemoryStream(input))
            {
                bmp = new Bitmap(ms);
            }
            return bmp;
        }

        public static TFTensor TensorFromBitmap(Bitmap image, ImageDims dims)
        {
            int INPUT_SIZE = dims.InputSize;
            int IMAGE_MEAN = dims.ImageMean;
            float IMAGE_STD = dims.ImageStd;

            var bitmap = new Bitmap(image, INPUT_SIZE, INPUT_SIZE);
            
            Color[] colors = new Color[bitmap.Size.Width * bitmap.Size.Height];
            
            int z = 0;
            for (int y = bitmap.Size.Height -1; y >= 0; y--) {
                for (int x = 0; x < bitmap.Size.Width; x++) {
                    colors[z] = bitmap.GetPixel(x, y);
                    z++;
                }
            }

            float[] floatValues = new float[(INPUT_SIZE * INPUT_SIZE) * 3];
            for (int i = 0; i < colors.Length; i++) {
                var color = colors[i];

                floatValues[i * 3] = (color.R - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 1] = (color.G - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 2] = (color.B - IMAGE_MEAN) / IMAGE_STD;
            }

            TFShape shape = new TFShape(1, INPUT_SIZE, INPUT_SIZE, 3);
            return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
        }
    }

    public class CoachResult
    {
        ///<summary>
        //Unsorted prediction results
        ///</summary>
        public List<LabelProbability> Results { get; private set; }
        
        ///<summary>
        //Sorted prediction results, descending in Confidence
        ///</summary>
        public List<LabelProbability> SortedResults { get; private set; }

        public CoachResult(string[] labels, float[] probabilities)
        {
            Results = new List<LabelProbability>();
            
            for (var i = 0; i < labels.Length; i++) {
                string label = labels[i];
                float probability = probabilities[i];
                
                Results.Add(new LabelProbability() {
                    Label = label,
                    Confidence = probability
                });
            }
            SortedResults = Results.OrderByDescending(r => r.Confidence).ToList();
        }

        ///<summary>
        ///Most Confident result
        ///</summary>
        public LabelProbability Best()
        {
            return SortedResults.FirstOrDefault();
        }

        ///<summary>
        ///Least Confident result
        ///</summary>
        public LabelProbability Worst()
        {
            return SortedResults.LastOrDefault();
        }
    }

    public struct LabelProbability
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
    }

    public class CoachModel {
        private readonly float COACH_VERSION = 2f;

        private TFGraph Graph { get; set; }
        private TFSession Session { get; set; }
        private string[] Labels { get; set; }
        private ImageDims ImageDims { get; set; }

        ///<summary>
        ///<param>Model graph</param>
        ///<param>Model labels</param>
        ///<param>Base module used for training</param>
        ///<param>Model SDK version</param>
        ///</summary>
        public CoachModel(TFGraph graph, string[] labels, string module, float coachVersion) {
            if (COACH_VERSION != coachVersion) {
                throw new Exception($"Coach model v{coachVersion} incompatible with SDK version {COACH_VERSION}");
            }

            this.Graph = graph;
            this.Labels = labels;

            this.Session = new TFSession(this.Graph);
            
            int size = int.Parse(module.Substring(module.Length-3, 3));
            this.ImageDims = new ImageDims(size, 0, 255);
        }
        
        private TFTensor ReadTensorFromBytes(byte[] image) {
            var bmp = ImageUtil.BitmapFromBytes(image);
            return ImageUtil.TensorFromBitmap(bmp, this.ImageDims);
        }

        private TFTensor ReadTensorFromFile(string filePath) {
            var bmp = ImageUtil.BitmapFromFile(filePath);
            return ImageUtil.TensorFromBitmap(bmp, this.ImageDims);
        }

        ///<summary>
        ///Parses the specified image as a Tensor and runs it through the loaded model
        ///<param>Path to the sample image</param>
        ///<param>Name of the input in the graph</param>
        ///<param>Name of the output in the graph</param>
        ///</summary>
        public CoachResult Predict(string image, string inputName = "input", string outputName = "output") {
            var imageTensor = ReadTensorFromFile(image);
            return GetGraphResult(imageTensor, inputName, outputName);
        }

        ///<summary>
        ///Parses the specified image bytes as a Tensor and runs it through the loaded model
        ///<param>Image as byte array</param>
        ///<param>Name of the input in the graph</param>
        ///<param>Name of the output in the graph</param>
        ///</summary>
        public CoachResult Predict(byte[] image, string inputName = "input", string outputName = "output") {
            var imageTensor = ReadTensorFromBytes(image);
            return GetGraphResult(imageTensor, inputName, outputName);
        }

        private CoachResult GetGraphResult(TFTensor imageTensor, string inputName = "input", string outputName = "output") {
            var runner = Session.GetRunner();

            var gInput = this.Graph[inputName];
            var gResult = this.Graph[outputName];

            runner.AddInput(gInput[0], imageTensor);
            runner.Fetch(gResult[0]);

            var result = runner.Run()[0];
            var resultShape = result.Shape;

            if (result.NumDims != 2 || resultShape[0] != 1)
            {
                var shape = "";
                foreach (var d in resultShape)
                {
                    shape += $"{d} ";
                }
                shape = shape.Trim();
            }

            var probabilities = ((float[][])result.GetValue(jagged: true))[0];
            return new CoachResult(Labels, probabilities);
        }
    }

    public struct Model {
        public string name { get; set; }
        public string status { get; set; }
        public int version { get; set; }
        public string module { get; set; }
        public string[] labels { get; set; }
        public float coachVersion { get; set; }
    }

    public class Profile {
        [JsonProperty("bucket")]
        public string Bucket { get; set; }

        [JsonProperty("models")]
        public Model[] Models { get; set; }
    }

    public enum ModelType {
        Frozen, Unity, Mobile
    }

    public class CoachClient {
        
        private bool IsDebug { get; set; }
        private Profile Profile { get; set; }
        private string ApiKey { get; set; }

        ///<summary>
        ///<para>If true, additional logs will be displayed</para>
        ///</summary>
        public CoachClient(bool isDebug = false) {
            this.IsDebug = isDebug;
        }

        ///<summary>
        ///Authenticates with Coach service and allows for model caching. Accepts API Key as its only parameter
        ///<param>Your API key</param>
        ///</summary>
        public async Task<CoachClient> Login(string apiKey) {
            if (apiKey == String.Empty) {
                throw new Exception("Invalid API Key");
            }
            this.ApiKey = apiKey;
            this.Profile = await GetProfile();

            return this;
        }

        private bool IsAuthenticated() {
            return this.Profile != null;
        }

        private Model ReadManifest(string path) {
            return JsonConvert.DeserializeObject<Model>(File.ReadAllText(path));
        }

        private async Task<Profile> GetProfile() {
            var id = this.ApiKey.Substring(0, 5);
            var url = $"https://x27xyu10z1.execute-api.us-east-1.amazonaws.com/latest/profile?id={id}";

            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);

            var response = await request.GetAsync(url);
            response.EnsureSuccessStatusCode();

            string responseBody = await response.Content.ReadAsStringAsync();
            var profile = JsonConvert.DeserializeObject<Profile>(responseBody);
            return profile;
        }

        ///<summary>
        ///Downloads model from Coach service to disk
        ///<param>Name of model</param>
        ///<param>Path to cache model</param>
        ///<param>If true, the download will be skipped if the model of the same filename already exists</param>
        ///<param>The type of model to be cached</param>
        ///</summary>
        public async Task CacheModel(string modelName, string path = ".", bool skipMatch = true, ModelType modelType = ModelType.Frozen) {
            if (!IsAuthenticated())
                throw new Exception("User is not authenticated");

            Model model = this.Profile.Models.Single(m => m.name == modelName);
            int version = model.version;

            string modelDir = Path.Join(path, modelName);
            string profileManifest = Path.Join(modelDir, "manifest.json");

            if (File.Exists(profileManifest)) {
                // Load existing model manifest
                Model manifest = ReadManifest(profileManifest);

                int manifestVersion = manifest.version;
                int profileVersion = model.version;

                if (profileVersion == manifestVersion && skipMatch) {
                    if (this.IsDebug) {
                        Console.WriteLine("Version match, skipping model download");
                    }
                    return;
                }
            } else if (!Directory.Exists(modelDir)) {
                Directory.CreateDirectory(modelDir);
            }
            
            // Write downloaded manifest
            var json = JsonConvert.SerializeObject(model);
            File.WriteAllText(profileManifest, json);

            var baseUrl = $"https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{this.Profile.Bucket}/model-bin?object=trained/{modelName}/{version}/model";
            
            string modelFile = String.Empty;
            if (modelType == ModelType.Frozen)
            {
                modelFile = "frozen.pb";
            } else if (modelType == ModelType.Unity)
            {
                modelFile = "unity.bytes";

            } else if (modelType == ModelType.Mobile)
            {
                modelFile = "mobile.tflite";
            }
            var modelUrl = $"{baseUrl}/{modelFile}";

            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);
            request.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/octet-stream"));
            
            var requestMessage = new HttpRequestMessage(HttpMethod.Get, modelUrl);
            requestMessage.Content = new StringContent(String.Empty, Encoding.UTF8, "application/octet-stream");
            
            var response = await request.SendAsync(requestMessage);
            response.EnsureSuccessStatusCode();

            byte[] modelBytes = await response.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes(Path.Join(path, modelName, modelFile), modelBytes);
        }
        
        ///<summary>
        ///Loads model into memory
        ///<param>Path to the model</param>
        ///</summary>
        public CoachModel GetModel(string path) {
            var graphPath = Path.Join(path, "frozen.pb");
            var labelPath = Path.Join(path, "manifest.json");
            
            var graphBytes = File.ReadAllBytes(graphPath);

            // Load the graphdef
            var graph = new TFGraph();
            graph.Import(graphBytes);

            var manifest = ReadManifest(labelPath);
            
            string[] labels = manifest.labels;
            string baseModule = manifest.module;
            float coachVersion = manifest.coachVersion;

            return new CoachModel(graph, labels, baseModule, coachVersion);
        }
    
        ///<summary>
        ///Downloads model from Coach service to disk, and loads it into memory
        ///<param>Name of model</param>
        ///<param>Path to cache the model</param>
        ///</summary>
        public async Task<CoachModel> GetModelRemote(string modelName, string path=".") {
            await CacheModel(modelName, path);
            return GetModel(Path.Join(path, modelName));
        }
    }
}
