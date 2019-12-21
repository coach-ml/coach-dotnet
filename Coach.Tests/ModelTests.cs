using Coach;
using System;
using Xunit;

namespace Coach.Tests
{
    public class ModelTests
    {
        private readonly CoachModel _model;

        public ModelTests()
        {
            var c = new CoachClient(isDebug: true).Login("A2botdrxAn68aZh8Twwwt2sPBJdCfH3zO02QDMt0").Result;
            _model = c.GetModelRemote("small_flowers").Result;
        }

        [Fact]
        public void Predict()
        {
            var preciction = _model.Predict("rose.jpg");
            Assert.Equal("rose", preciction.Best().Label);
        }
    }
}
