import 'dart:async' show Future;
import 'package:flutter/services.dart' show rootBundle;
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_maps/maps.dart';
import 'package:syncfusion_flutter_sliders/sliders.dart';
import 'package:http/http.dart' as http;

class Home extends StatefulWidget {
  const Home({ Key? key }) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  Map<String,List<Country>>? _data = {};
  Map<String, MapShapeSource>? _dataSource = {};
  double val = 1870;
  Map? rawData = {};
  late final Future? getDataFuture;
  Map<String, List<MapColorMapper>>? colorMappers = {};

  Future<Map> fetchData(key) async {
    final response = await http.get(Uri.parse('https://faazabidi.github.io/CI-SOM/JSON-Data/CO2_map${key}.json'));

    if (response.statusCode == 200) {
      // If the server did return a 200 OK response,
      // then parse the JSON.
      return jsonDecode(response.body);
    } else {
      // If the server did not return a 200 OK response,
      // then throw an exception.
      throw Exception('Failed to load album');
    }
  }

  Future<Map> loadJsonData() async { 
    Map<dynamic, dynamic>? data;
    print("getting ");
    for (int x = 1870; x < 2021; x++) {
        print(x);
        // data = json.decode(await rootBundle.loadString('CO2_map' + x.toString() + '.json'));
        data = await fetchData(x.toString());
        rawData![x.toString()] = data;
        // print(rawData);
        _data![x.toString()] = loadModelsList(data);
        // print()
        colorMappers![x.toString()] = loadColorMappers(_data![x.toString()]!);
      }
    // var jsonText = await rootBundle.loadString('CO2_map$val.json'); 
    // data = json.decode(jsonText); 
    // 
    print("Success");
    
    // return 'success';
    return data!;
  }

  loadModelsList(raw) {
    List<Country> models = [];
    raw.keys.forEach((element) {
      Country temp = Country(element, List<int>.from(raw[element]));
      models.add(temp);
    });
    print("Models done");
    return models;
  }

  loadColorMappers(List<Country> _element_data) {
    List<MapColorMapper> all = [];
    for (int x = 0; x < _element_data.length; x++) {
      all.add(MapColorMapper(
        value: _element_data[x].country,
        color: Color.fromRGBO(_element_data[x].rgb[0], _element_data[x].rgb[1], _element_data[x].rgb[2], 1)
      ));
    }
    print("Mappers Done");
    return all;
  }

  @override
  void initState() {
    getDataFuture = loadJsonData();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double height = MediaQuery.of(context).size.height;

    return SafeArea(
      child: Scaffold(
        backgroundColor: const Color.fromARGB(255, 222, 222, 222),
        body: Center(
        child: Padding(
          padding: EdgeInsets.symmetric(vertical: height*0.05),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
            const Flexible(child: Text("SOM Visualization for Global Carbon Dioxide Emissions", style: TextStyle(color: Color.fromARGB(255, 43, 42, 42), fontSize: 28),)),
              SizedBox(height: height*0.08),
              FutureBuilder(
                future: getDataFuture,
                builder: (context, snap) {
                  if (snap.hasData) {
                    return Container(
                      height: height*0.65,
                      width: width*0.9,
                      child: SfMaps(
                        layers: [
                          MapShapeLayer(source: MapShapeSource.network(
                          'https://faazabidi.github.io/CI-SOM/world.geojson',
                          shapeDataField: 'ADM0_A3',
                          dataCount: _data![val.toString()]!.length,
                          primaryValueMapper: (int index) => _data![val.toString()]![index].country,
                          shapeColorValueMapper: (int index) => _data![val.toString()]![index].country,
                          shapeColorMappers: colorMappers![val.toString()]
                          ),
                          showDataLabels: false,
                          color:  const Color.fromARGB(255, 81, 81, 81),
                          strokeColor: const Color.fromARGB(255, 184, 184, 184),
                          ),
                        ],
                      ),
                  );
                  }

                  return Column(
                    children: [
                      const CircularProgressIndicator(color: Color.fromARGB(255, 48, 48, 48),),
                      SizedBox(height: height*0.03),
                      Text("Please wait. Fetching data...", style: TextStyle(color: Color.fromARGB(255, 43, 42, 42), fontSize: 18),)
                    ],
                  );
                  
                }
              ),
              SizedBox(height: height*0.03,),
            Flexible(child: Text("World in $val", style: const TextStyle(color: Color.fromARGB(255, 36, 36, 36), fontSize: 18),)),
            Padding(
              padding:  EdgeInsets.symmetric(horizontal: width*0.07),
              child: SfSlider(
                      min: 1870,
                      max: 2020,
                      value: val,
                      interval: 10,
                      showTicks: true,
                      showLabels: true,
                      enableTooltip: false,
                      minorTicksPerInterval: 1,
                      onChanged: (dynamic value){
                        // getDataFuture = loadJsonData();
                        setState(() {
                          val = value~/1;
                        });
                      },
                      activeColor: const Color.fromARGB(255, 45, 45, 45),
                      inactiveColor: const Color.fromARGB(255, 184, 184, 184),
                      
                    ),
            ),
            ],
          ),
        ),
        ),
      )
    );
  }
}


class Country {
  final String country;
  final List<int> rgb;

  Country(this.country, this.rgb);

}