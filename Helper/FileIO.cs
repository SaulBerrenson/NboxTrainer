using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NboxTrainer.ML;


namespace NboxTrainer.Helper
{
    internal static class FileIO
    {
        public static bool dirExist(string dir) => Directory.Exists(dir);

        public static IEnumerable<string> getListNamesDirectiories(string dir, SearchOption option = SearchOption.TopDirectoryOnly)
        {
            try
            {
                if (!dirExist(dir)) return new string[0];
                return Directory.EnumerateDirectories(dir, "*.*", option).Select(_dir => new DirectoryInfo(_dir).Name);
            }
            catch (Exception e)
            {
                return new string[0];
            }
        }


        public static IEnumerable<string> getListDirectiories(string dir, SearchOption option = SearchOption.TopDirectoryOnly)
        {
            try
            {
                if (!dirExist(dir)) return new string[0];
                return Directory.EnumerateDirectories(dir, "*.*", option);
            }
            catch (Exception e)
            {
                return new string[0];
            }
        }


        public static int countFiles(string dir, string pattern = "*.*", SearchOption option = SearchOption.TopDirectoryOnly)
        {
            int? count = Directory.EnumerateFiles(dir, pattern, option)?.Count();
            if (count > 0)
            {
                return (int)count;
            }
            return 0;
        }


        public static IEnumerable<ImageData> loadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData(file, label);
            }
        }

    }
}