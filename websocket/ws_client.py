
import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class SocketService {
  SocketService._();

  static final SocketService instance = SocketService._();

  WebSocketChannel? _channel;

  int? _userId;

  Timer? _reconnectTimer;

  bool _isReconnecting = false;
  bool _manualDisconnect = false;

  final StreamController<Map<String, dynamic>> _controller =
      StreamController<Map<String, dynamic>>.broadcast();

  bool get isConnected => _channel != null;

  Stream<Map<String, dynamic>> get stream => _controller.stream;

  Future<void> connect(int userId) async {
    if (_channel != null) {
      return;
    }

    _userId = userId;
    _manualDisconnect = false;

    try {
      // .env에서 API 서버 주소 가져오기
      final String apiBaseUrl = dotenv.env['API_BASE_URL'] ?? '';

      if (apiBaseUrl.isEmpty) {
        throw Exception('API_BASE_URL이 .env에 설정되지 않았습니다.');
      }

      // HTTP 주소를 WebSocket 주소로 변환
      final String wsUrl = apiBaseUrl
          .replaceFirst('http://', 'ws://')
          .replaceFirst('https://', 'wss://');

      debugPrint('[WS] 연결 시도: $wsUrl');

      final channel = WebSocketChannel.connect(
        Uri.parse(wsUrl),
      );

      await channel.ready;

      if (_manualDisconnect) {
        await channel.sink.close();
        return;
      }

      _channel = channel;

      _reconnectTimer?.cancel();
      _reconnectTimer = null;
      _isReconnecting = false;

      debugPrint('[WS] WebSocket 연결 성공');

      // 서버에 사용자 등록
      send({
        'type': 'registerUser',
        'user_id': userId,
      });

      // 서버에서 오는 메시지는 여기서 한 번만 받음
      channel.stream.listen(
        (message) {
          try {
            final decoded = jsonDecode(message);

            if (decoded is Map<String, dynamic>) {

              // ======================================
              // 수신 메시지 타입 확인
              // ======================================
              debugPrint(
                '[WS] 메시지 수신 type: ${decoded['type']}',
              );

              // ======================================
              // SOM 활동성 분석 결과 확인
              // ======================================
              if (decoded['type'] == 'activity_result') {
                debugPrint(
                  '[FLUTTER] 활동성 결과 수신: $decoded',
                );

                final data = decoded['data'];

                if (data is Map) {
                  debugPrint(
                    '[FLUTTER] 건강=${data['healthy']}, '
                    '주의=${data['abnormal']}',
                  );
                } else {
                  debugPrint(
                    '[FLUTTER] 활동성 data 형식 오류: $data',
                  );
                }
              }

              // ======================================
              // 어종 분석 결과 확인
              // ======================================
              if (decoded['type'] == 'species_result') {
                debugPrint(
                  '[FLUTTER] 어종 분석 결과 수신: $decoded',
                );
              }

              // 기존 메시지 전달 유지
              _controller.add(decoded);
            } else {
              debugPrint(
                '[WS] 예상하지 못한 데이터 형식: $decoded',
              );
            }
          } catch (e) {
            debugPrint('WebSocket 데이터 변환 실패: $e');
          }
        },
        onError: (error) {
          debugPrint('[WS] WebSocket 오류: $error');

          if (_channel == channel) {
            _channel = null;
          }

          _reconnect();
        },
        onDone: () {
          debugPrint('[WS] WebSocket 연결 종료됨');

          if (_channel == channel) {
            _channel = null;
          }

          _reconnect();
        },
      );
    } catch (error) {
      debugPrint('[WS] WebSocket 연결 실패: $error');

      _channel = null;

      _reconnect();
    }
  }

  void _reconnect() {
    // 사용자가 직접 disconnect 한 경우
    if (_manualDisconnect) {
      return;
    }

    // 이미 재연결 대기 중
    if (_isReconnecting) {
      return;
    }

    // userId가 없으면 재연결 불가능
    final userId = _userId;

    if (userId == null) {
      return;
    }

    _isReconnecting = true;

    debugPrint('[WS] 3초 후 WebSocket 재연결 시도');

    _reconnectTimer?.cancel();

    _reconnectTimer = Timer(
      const Duration(seconds: 3),
      () async {
        _isReconnecting = false;

        if (_manualDisconnect) {
          return;
        }

        debugPrint('[WS] WebSocket 재연결 중...');

        await connect(userId);
      },
    );
  }

  void send(Map<String, dynamic> data) {
    final channel = _channel;

    if (channel == null) {
      debugPrint('[WS] WebSocket이 연결되어 있지 않습니다.');
      return;
    }

    channel.sink.add(
      jsonEncode(data),
    );

    debugPrint('[WS] 전송: $data');
  }

  Future<void> disconnect() async {
    _manualDisconnect = true;

    _reconnectTimer?.cancel();
    _reconnectTimer = null;

    _isReconnecting = false;

    final channel = _channel;

    _channel = null;
    _userId = null;

    if (channel != null) {
      await channel.sink.close();
    }

    debugPrint('[WS] WebSocket 연결 종료');
  }
}
