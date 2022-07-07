

// class LoadData{
//   static const String file_path = 'data.json';

//   static Future<Map<String, dynamic>> load() async {
//     final String path = join(await _getDataFolder(), _kDataFile);
//     return json.decode(await new File(path).readAsString());
//   }

//   static Future<String> _getDataFolder() async {
//     final String path = join(await getApplicationDocumentsDirectory().then((d) => d.path), _kDataFolder);
//     if (!(await new Directory(path).exists())) {
//       await new Directory(path).create(recursive: true);
//     }
//     return path;
//   }
// }