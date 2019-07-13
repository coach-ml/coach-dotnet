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

    public struct CoachResult
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
    }

    public class CoachModel {
        private TFGraph Graph { get; set; }
        private TFSession Session { get; set; }
        private string[] Labels { get; set; }
        private ImageDims ImageDims { get; set; }
        public CoachModel(TFGraph graph, string[] labels, string module) {
            this.Graph = graph;
            this.Labels = labels;

            this.Session = new TFSession();
            
            int size = int.Parse(module.Substring(module.Length-3, module.Length));
            this.ImageDims = new ImageDims(size, 0, 1);
        }
        
        private TFTensor ReadTensorFromBytes(byte[] image) {
            var bmp = ImageUtil.BitmapFromBytes(image);
            return ImageUtil.TensorFromBitmap(bmp, this.ImageDims);
        }

        private TFTensor ReadTensorFromFile(string filePath) {
            var bmp = ImageUtil.BitmapFromFile(filePath);
            return ImageUtil.TensorFromBitmap(bmp, this.ImageDims);
        }

        public CoachResult Predict(string image) {
            var imageTensor = ReadTensorFromFile(image);
            return GetGraphResult(imageTensor);
        }

        public CoachResult Predict(byte[] image) {
            var imageTensor = ReadTensorFromBytes(image);
            return GetGraphResult(imageTensor);
        }

        private CoachResult GetGraphResult(TFTensor imageTensor) {
            var inputName = "lambda_input";
            var outputName = "output/Softmax";

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
            int bestIdx = 0;
            float best = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                if (probabilities[i] > best)
                {
                    bestIdx = i;
                    best = probabilities[i];
                }
            }

            var bestMatch = new CoachResult()
            {
                Label = this.Labels[bestIdx],
                Confidence = best
            };

            return bestMatch;
        }
    }

    public struct Model {
        public string name { get; set; }
        public string status { get; set; }
        public int version { get; set; }
        public string module { get; set; }
        public string[] labels { get; set; }
    }

    public class Profile {
        [JsonProperty("id")]
        public string Id { get; set; }
        [JsonProperty("bucket")]
        public string Bucket { get; set; }

        [JsonProperty("models")]
        public Model[] Models { get; set; }
    }

    public class CoachClient {
        
        public bool IsDebug { get; private set; }
        private Profile Profile { get; set; }
        private string ApiKey { get; set; }

        public CoachClient(bool isDebug = false) {
            this.IsDebug = isDebug;
        }

        public async Task Login(string apiKey) {
            this.ApiKey = apiKey;
            this.Profile = await GetProfile();
        }

        private bool IsAuthenticated() {
            return this.Profile != null;
        }

        private Model ReadManifest(string path) {
            return JsonConvert.DeserializeObject<Model>(File.ReadAllText(path));
        }

        public async Task CacheModel(string name, string path=".") {
            if (!IsAuthenticated())
                throw new Exception("User is not authenticated");

            // Create the target dir
            if (!Directory.Exists(name))
                Directory.CreateDirectory(name);

            Model model = this.Profile.Models.Single(m => m.name == name);

            int profileVersion = model.version;
            string profileManifest = $"{path}/{name}/manifest.json";

            if (File.Exists(profileManifest)) {
                // Load existing model manifest
                Model manifest = ReadManifest(profileManifest);
                int manifestVersion = manifest.version;

                if (profileVersion == manifestVersion) {
                    if (this.IsDebug) {
                        Console.WriteLine("Version match, skipping model download");
                    }

                    return;
                }
            } else {
                var json = JsonConvert.SerializeObject(model);
                File.WriteAllText(profileManifest, json);
            }

            var baseUrl = $"https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{this.Profile.Bucket}/model-bin?object=trained/{name}/{profileVersion}/model";
                    
            var modelFile = "frozen.pb";
            var modelUrl = $"{baseUrl}/{modelFile}";


            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);
            request.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/octet-stream"));
            
            var requestMessage = new HttpRequestMessage(HttpMethod.Get, modelUrl);
            requestMessage.Content = new StringContent(String.Empty, Encoding.UTF8, "application/octet-stream");
            
            var response = await request.SendAsync(requestMessage);
            response.EnsureSuccessStatusCode();

            byte[] modelBytes = await response.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes($"{path}/{name}/{modelFile}", modelBytes);
        }

        private async Task<Profile> GetProfile() {
            var id = this.ApiKey.Substring(0, 5);
            var url = $"https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/{id}";

            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);

            var response = await request.GetAsync(url);
            response.EnsureSuccessStatusCode();

            string responseBody = await response.Content.ReadAsStringAsync();
            var profile = JsonConvert.DeserializeObject<Profile>(responseBody);
            return profile;
        }

        public CoachModel GetModel(string path) {
            var graphPath = $"{path}/frozen.pb";
            var labelPath = $"{path}/manifest.json";
            
            var graphBytes = File.ReadAllBytes(graphPath);

            // Load the graphdef
            var graph = new TFGraph();
            graph.Import(graphBytes);

            var manifest = ReadManifest(labelPath);
            
            string[] labels = manifest.labels;
            string baseModule = manifest.module;

            return new CoachModel(graph, labels, baseModule);
        }
    }
}