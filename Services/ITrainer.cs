using System.Threading.Tasks;

namespace NboxTrainer.Services
{
    public interface ITrainer
    {
        public Task<bool> Train();
        public Task SaveModel();
    }
}