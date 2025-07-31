// Flutter web plugin registrant file.
//
// Generated file. Do not edit.
//

// @dart = 2.13
// ignore_for_file: type=lint

import 'package:awesome_notifications/awesome_notifications_web.dart';
import 'package:device_info_plus/src/device_info_plus_web.dart';
import 'package:flutter_blue_plus_web/flutter_blue_plus_web.dart';
import 'package:permission_handler_html/permission_handler_html.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';

void registerPlugins([final Registrar? pluginRegistrar]) {
  final Registrar registrar = pluginRegistrar ?? webPluginRegistrar;
  AwesomeNotificationsWeb.registerWith(registrar);
  DeviceInfoPlusWebPlugin.registerWith(registrar);
  FlutterBluePlusWeb.registerWith(registrar);
  WebPermissionHandler.registerWith(registrar);
  registrar.registerMessageHandler();
}
