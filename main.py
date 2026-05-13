from storage import get_cloud_storage
from bson import ObjectId
from pymongo import MongoClient
from datetime import datetime, time, timezone
import pytz  # ✅ Agregar esta dependencia: pip install pytz
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from dotenv import load_dotenv
import os
from typing import Optional, Dict, List, Any

load_dotenv()

# ✅ Configurar zona horaria de Perú
TIMEZONE = pytz.timezone('America/Lima')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
MONGODB_URI = os.getenv(
    'MONGODB_URI', 'mongodb+srv://carlosjesusordazhoyos_db_user:Z9sVzYmBdnKy5Y1i@cluster0.l4wjrmp.mongodb.net')
client = MongoClient(MONGODB_URI)
db = client['test']
users_collection = db['users']
asistencias_collection = db['asistencias']


# ✅ Nueva función helper para obtener hora actual en zona horaria local
def get_now_local() -> datetime:
    """Retorna datetime actual en zona horaria de Perú (UTC-5)"""
    return datetime.now(TIMEZONE)


# ✅ Nueva función para convertir UTC a local
def utc_to_local(dt: datetime) -> datetime:
    """Convierte datetime UTC (de MongoDB) a hora local de Perú"""
    if dt.tzinfo is None:
        # Si es naive, asumir que es UTC
        dt = pytz.utc.localize(dt)
    return dt.astimezone(TIMEZONE)


def get_expected_schedule_for_day(schedule_config: Dict, date: datetime) -> Optional[Dict]:
    """
    Obtiene el horario esperado para un día específico
    """
    if not schedule_config:
        return None

    # ✅ Asegurar que usamos la fecha en hora local
    if date.tzinfo is not None:
        local_date = date.astimezone(TIMEZONE)
    else:
        local_date = TIMEZONE.localize(date)

    day_names = ['monday', 'tuesday', 'wednesday',
                 'thursday', 'friday', 'saturday', 'sunday']
    day_of_week = local_date.weekday()  # 0 = Monday, 6 = Sunday
    day_name = day_names[day_of_week]

    week_schedule = schedule_config.get('weekSchedule', [])
    day_schedule = next(
        (d for d in week_schedule if d['day'] == day_name), None)

    if not day_schedule or not day_schedule.get('isWorkday', False):
        return None

    return {
        'day': day_name,
        'periods': [{'start': p['start'], 'end': p['end']} for p in day_schedule.get('periods', [])],
        'isWorkday': day_schedule.get('isWorkday', False),
        'totalHours': day_schedule.get('totalHours', 0)
    }


def is_remote_day(schedule_config: Optional[Dict], date: datetime) -> bool:
    if not schedule_config or not schedule_config.get('remoteDays'):
        return False

    if date.tzinfo is not None:
        local_date = date.astimezone(TIMEZONE)
    else:
        local_date = TIMEZONE.localize(date)

    day_names = ['monday', 'tuesday', 'wednesday',
                 'thursday', 'friday', 'saturday', 'sunday']
    return day_names[local_date.weekday()] in schedule_config.get('remoteDays', [])


def parse_time_string(time_str: str) -> time:
    """
    Convierte string de hora 'HH:MM' a objeto time
    """
    try:
        hours, minutes = map(int, time_str.split(':'))
        return time(hours, minutes)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Formato de hora inválido: {time_str}")


def calculate_schedule_compliance(
    entry_time: Optional[datetime],
    exit_time: Optional[datetime],
    expected_schedule: Optional[Dict],
    flexible_minutes: int,
    is_flexible: bool
) -> Dict:
    """
    Calcula el cumplimiento del horario basándose en el horario esperado
    """
    compliance = {
        'isLateEntry': False,
        'minutesLateEntry': 0,
        'isEarlyDeparture': False,
        'minutesEarlyDeparture': 0,
        'flexibleMinutesApplied': flexible_minutes,
        'wasFlexible': is_flexible
    }

    if not expected_schedule:
        return compliance

    if not expected_schedule.get('isWorkday', False):
        return compliance

    periods = expected_schedule.get('periods', [])
    if not periods:
        return compliance

    # ✅ Verificar entrada (primer período)
    if entry_time and len(periods) > 0:
        try:
            # ✅ Convertir entry_time de UTC a hora local
            local_entry = utc_to_local(entry_time)

            expected_start_str = periods[0]['start']
            expected_start_time = parse_time_string(expected_start_str)

            # ✅ Crear datetime en zona horaria local con la hora esperada
            expected_start = TIMEZONE.localize(datetime(
                local_entry.year,
                local_entry.month,
                local_entry.day,
                expected_start_time.hour,
                expected_start_time.minute,
                0,
                0
            ))

            # Calcular diferencia en minutos
            diff = (local_entry - expected_start).total_seconds() / 60

            # Aplicar margen de flexibilidad
            effective_margin = flexible_minutes if is_flexible else 0

            if diff > effective_margin:
                compliance['isLateEntry'] = True
                compliance['minutesLateEntry'] = int(diff - effective_margin)
        except Exception as e:
            print(f"Error calculando tardanza de entrada: {str(e)}")

    # ✅ Verificar salida (último período)
    if exit_time and len(periods) > 0:
        try:
            # ✅ Convertir exit_time de UTC a hora local
            local_exit = utc_to_local(exit_time)

            last_period = periods[-1]
            expected_end_str = last_period['end']
            expected_end_time = parse_time_string(expected_end_str)

            # ✅ Crear datetime en zona horaria local con la hora esperada
            expected_end = TIMEZONE.localize(datetime(
                local_exit.year,
                local_exit.month,
                local_exit.day,
                expected_end_time.hour,
                expected_end_time.minute,
                0,
                0
            ))

            # Calcular diferencia en minutos (salida temprana)
            diff = (expected_end - local_exit).total_seconds() / 60

            # Aplicar margen de flexibilidad
            effective_margin = flexible_minutes if is_flexible else 0

            if diff > effective_margin:
                compliance['isEarlyDeparture'] = True
                compliance['minutesEarlyDeparture'] = int(
                    diff - effective_margin)
        except Exception as e:
            print(f"Error calculando salida temprana: {str(e)}")

    return compliance


def get_user_schedule_config(user_id: str) -> Optional[Dict]:
    """
    Obtiene la configuración de horario activa del usuario
    """
    try:
        schedule_configs_collection = db['scheduleconfigs']
        schedule_config = schedule_configs_collection.find_one({
            'userId': ObjectId(user_id),
            'active': True
        })
        return schedule_config
    except Exception as e:
        print(f"Error al obtener configuración de horario: {str(e)}")
        return None


def image_to_numpy(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400, detail="No se pudo procesar la imagen")
    return img


def get_face_embedding(img: np.ndarray):
    faces = face_app.get(img)
    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No se detectó ningún rostro en la imagen"
        )
    if len(faces) > 1:
        raise HTTPException(
            status_code=400,
            detail="Se detectaron múltiples rostros. Por favor, tome una foto con un solo rostro visible"
        )
    return faces[0].embedding


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray, threshold=0.4):
    similarity = np.dot(emb1, emb2) / \
        (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity > threshold, float(similarity)


def calcular_distancia(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * \
        cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def validar_ubicacion(lat_usuario: float, lon_usuario: float, lat_sede: float, lon_sede: float, radio: float):
    distancia = calcular_distancia(
        lat_usuario, lon_usuario, lat_sede, lon_sede)
    valido = distancia <= radio

    return {
        "valido": valido,
        "distancia": round(distancia, 2)
    }


def get_range_date():
    """✅ Retorna el rango de fechas del día actual en hora local"""
    now = get_now_local()
    # Crear inicio y fin del día en hora local
    start = TIMEZONE.localize(datetime(now.year, now.month, now.day, 0, 0, 0))
    end = TIMEZONE.localize(datetime(now.year, now.month, now.day, 23, 59, 59))

    # ✅ Convertir a UTC para consultar MongoDB
    start_utc = start.astimezone(pytz.utc).replace(tzinfo=None)
    end_utc = end.astimezone(pytz.utc).replace(tzinfo=None)

    return start_utc, end_utc


def serialize_attendance(attendance: Optional[Dict]) -> Optional[Dict]:
    if not attendance:
        return None

    return {
        "entrada": utc_to_local(attendance["entrada"]).isoformat() if attendance.get("entrada") else None,
        "salida": utc_to_local(attendance["salida"]).isoformat() if attendance.get("salida") else None,
        "horas_trabajadas": attendance.get("horas_trabajadas"),
        "valido_entrada": attendance.get("valido_entrada"),
        "valido_salida": attendance.get("valido_salida")
    }


@app.get("/")
async def root():
    return {
        "message": "API de Asistencia con Reconocimiento Facial",
        "version": "1.0.0",
        "status": "online"
    }


@app.post("/api/users/{user_id}/photo")
async def update_user_photo(
    user_id: str,
    photo: UploadFile = File(...)
):
    try:
        cloud_storage = get_cloud_storage()
        try:
            user = users_collection.find_one({"_id": ObjectId(user_id)})
        except:
            raise HTTPException(
                status_code=400, detail="ID de usuario inválido")

        if not user:
            raise HTTPException(
                status_code=404, detail="Usuario no encontrado")

        contents = await photo.read()
        await photo.seek(0)
        img = image_to_numpy(contents)
        embedding = get_face_embedding(img)
        if user.get("photo"):
            try:
                await cloud_storage.delete_file_by_url(user["photo"])
            except Exception as e:
                print(f"No se pudo eliminar la foto anterior: {str(e)}")
        photo_url = await cloud_storage.upload_from_file(photo, folder="usuarios")

        # ✅ Usar get_now_local() pero guardar como naive UTC en MongoDB
        now_utc = datetime.utcnow()

        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "photo": photo_url,
                    "embedding": embedding.tolist(),
                    "updatedAt": now_utc
                }
            }
        )

        return {
            "success": True,
            "message": "Foto actualizada exitosamente",
            "photo_url": photo_url
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error al actualizar foto: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la foto: {str(e)}"
        )


@app.post("/api/attendance/marcar")
async def marcar_asistencia(
    userId: str = Form(...),
    tipo: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    photo: UploadFile = File(...)
):
    try:
        if tipo not in ['entrada', 'salida']:
            raise HTTPException(
                status_code=400,
                detail="Tipo inválido. Use 'entrada' o 'salida'"
            )

        if False and (not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float))):
            raise HTTPException(
                status_code=400,
                detail="Las coordenadas deben ser números válidos"
            )

        if False and (latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180):
            raise HTTPException(
                status_code=400,
                detail="Coordenadas fuera de rango válido"
            )

        try:
            user = users_collection.find_one({"_id": ObjectId(userId)})
        except:
            raise HTTPException(
                status_code=400, detail="ID de usuario inválido")

        if not user:
            raise HTTPException(
                status_code=404, detail="Usuario no encontrado")

        if "embedding" not in user or not user["embedding"]:
            raise HTTPException(
                status_code=400,
                detail="El usuario no tiene una foto de perfil registrada. Por favor, suba una foto primero."
            )
        contents = await photo.read()
        img = image_to_numpy(contents)
        current_embedding = get_face_embedding(img)
        stored_embedding = np.array(user["embedding"])
        threshold = 0.4
        is_match, similarity = compare_embeddings(
            current_embedding, stored_embedding, threshold=threshold)

        if not is_match:
            return {
                "success": False,
                "verified": False,
                "message": "El rostro no coincide con el usuario registrado",
                "similarity": round(similarity, 4),
                "required_similarity": threshold
            }
        if False and not user.get("sede"):
            raise HTTPException(
                status_code=400,
                detail="No se ha asignado una sede al usuario"
            )

        try:
            sede = db['sedes'].find_one({"_id": ObjectId(user["sede"])}) if user.get("sede") else None
        except:
            raise HTTPException(
                status_code=400,
                detail="ID de sede inválido"
            )

        if False and not sede:
            raise HTTPException(
                status_code=400,
                detail="La sede asignada no existe"
            )

        if False and (not sede.get("latitude") or not sede.get("longitude") or not sede.get("radio")):
            raise HTTPException(
                status_code=400,
                detail="La sede no tiene coordenadas o radio configurados"
            )

        validacion = {
            "valido": True,
            "distancia": None
        }

        schedule_config = get_user_schedule_config(userId)
        ahora_utc = datetime.utcnow()
        ahora_local = get_now_local()
        is_remote_workday = is_remote_day(schedule_config, ahora_local)

        if is_remote_workday:
            validacion = {
                "valido": True,
                "distancia": None,
                "radio_permitido": None,
                "omitida_por_remoto": True
            }
            sede = None
        else:
            if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail="Las coordenadas son obligatorias en dias presenciales"
                )

            if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
                raise HTTPException(
                    status_code=400,
                    detail="Coordenadas fuera de rango vÃ¡lido"
                )

            if not user.get("sede"):
                raise HTTPException(
                    status_code=400,
                    detail="No se ha asignado una sede al usuario"
                )

            try:
                sede = db['sedes'].find_one({"_id": ObjectId(user["sede"])})
            except:
                raise HTTPException(
                    status_code=400,
                    detail="ID de sede invÃ¡lido"
                )

            if not sede:
                raise HTTPException(
                    status_code=400,
                    detail="La sede asignada no existe"
                )

            if not sede.get("latitude") or not sede.get("longitude") or not sede.get("radio"):
                raise HTTPException(
                    status_code=400,
                    detail="La sede no tiene coordenadas o radio configurados"
                )

            validacion = validar_ubicacion(
                latitude,
                longitude,
                float(sede["latitude"]),
                float(sede["longitude"]),
                float(sede["radio"])
            )

        expected_schedule = None
        schedule_config_snapshot = None
        if schedule_config:
            expected_schedule = get_expected_schedule_for_day(
                schedule_config, ahora_local)
            schedule_config_snapshot = {
                'scheduleConfigId': schedule_config['_id'],
                'configName': schedule_config.get('name', ''),
                'configColor': schedule_config.get('color', '#FFC0CB'),
                'remoteDays': schedule_config.get('remoteDays', [])
            }

        start, end = get_range_date()
        asistencia = asistencias_collection.find_one({
            "user": ObjectId(userId),
            "createdAt": {"$gte": start, "$lt": end}
        })
        if tipo == 'entrada':
            if asistencia:
                return {
                    "success": False,
                    "error": "La asistencia de entrada ya fue marcada hoy",
                    "asistencia_existente": {
                        "entrada": asistencia.get("entrada").isoformat() if asistencia.get("entrada") else None,
                        "salida": asistencia.get("salida").isoformat() if asistencia.get("salida") else None
                    }
                }

            # ✅ Calcular cumplimiento de horario en la entrada
            flexible_minutes = schedule_config.get(
                'flexibleMinutes', 30) if schedule_config else 30
            is_flexible = schedule_config.get(
                'isFlexible', True) if schedule_config else True

            schedule_compliance = calculate_schedule_compliance(
                ahora_utc,  # ✅ Pasamos UTC, la función lo convertirá internamente
                None,
                expected_schedule,
                flexible_minutes,
                is_flexible
            )

            nueva_asistencia = {
                "entrada": ahora_utc,
                "user": ObjectId(userId),
                "sede": ObjectId(user["sede"]) if user.get("sede") else None,
                "latitude_entrada": latitude,
                "longitude_entrada": longitude,
                "valido_entrada": validacion["valido"],
                "similarity_entrada": similarity,

                # Snapshot del horario
                "expectedSchedule": expected_schedule,
                "scheduleCompliance": schedule_compliance,
                "scheduleConfigSnapshot": schedule_config_snapshot,

                "createdAt": ahora_utc,
                "updatedAt": ahora_utc
            }

            result = asistencias_collection.insert_one(nueva_asistencia)
            attendance_response = serialize_attendance(nueva_asistencia)

            return {
                "success": True,
                "verified": True,
                "tipo": "entrada",
                "message": "Entrada registrada correctamente" if validacion["valido"] else "Entrada registrada fuera de rango",
                "user": {
                    "nombre": f"{user.get('name', '')} {user.get('lname', '')}".strip(),
                    "dni": user.get("dni"),
                    "cargo": user.get("position")
                },
                "validacion_ubicacion": {
                    "valido": validacion["valido"],
                    "distancia": validacion["distancia"],
                    "radio_permitido": float(sede["radio"]) if sede else None,
                    "omitida_por_remoto": validacion.get("omitida_por_remoto", False)
                },
                "validacion_facial": {
                    "similarity": similarity,
                    "threshold": threshold
                },
                # Información de cumplimiento
                "schedule_compliance": {
                    "is_late_entry": schedule_compliance['isLateEntry'],
                    "minutes_late_entry": schedule_compliance['minutesLateEntry']
                } if expected_schedule else None,
                "asistencia_id": str(result.inserted_id),
                "attendance": attendance_response,
                "timestamp": ahora_local.isoformat()  # ✅ Retornar hora local al cliente
            }

        # ====== MANEJO DE SALIDA ======
        if tipo == 'salida':
            if not asistencia:
                raise HTTPException(
                    status_code=400,
                    detail="No se ha marcado una hora de entrada hoy"
                )

            if not asistencia.get("entrada"):
                raise HTTPException(
                    status_code=400,
                    detail="No existe un registro de entrada válido"
                )

            if asistencia.get("salida"):
                return {
                    "success": False,
                    "error": "La salida ya fue registrada hoy",
                    "asistencia_existente": {
                        "entrada": asistencia.get("entrada").isoformat() if asistencia.get("entrada") else None,
                        "salida": asistencia.get("salida").isoformat() if asistencia.get("salida") else None,
                        "horas_trabajadas": asistencia.get("horas_trabajadas")
                    }
                }

            salida_utc = datetime.utcnow()
            salida_local = get_now_local()

            diff_ms = (salida_utc - asistencia["entrada"]).total_seconds()
            horas_trabajadas = round(max(0, diff_ms) / 3600, 2)

            # ✅ Usar el snapshot guardado en la entrada
            stored_expected_schedule = asistencia.get('expectedSchedule')
            stored_config_snapshot = asistencia.get('scheduleConfigSnapshot')
            stored_compliance = asistencia.get('scheduleCompliance', {})

            # ✅ Recalcular cumplimiento completo (entrada + salida)
            schedule_compliance = calculate_schedule_compliance(
                asistencia["entrada"],
                salida_utc,
                stored_expected_schedule,
                stored_compliance.get('flexibleMinutesApplied', 30),
                stored_compliance.get('wasFlexible', True)
            )

            update_data = {
                "latitude_salida": latitude,
                "longitude_salida": longitude,
                "salida": salida_utc,
                "valido_salida": validacion["valido"],
                "similarity_salida": similarity,
                "horas_trabajadas": horas_trabajadas,
                "scheduleCompliance": schedule_compliance,
                "updatedAt": salida_utc
            }

            # ✅ Si no existía snapshot en la entrada, agregarlo ahora
            if not stored_expected_schedule and expected_schedule:
                update_data["expectedSchedule"] = expected_schedule
            if not stored_config_snapshot and schedule_config_snapshot:
                update_data["scheduleConfigSnapshot"] = schedule_config_snapshot

            asistencias_collection.update_one(
                {"_id": asistencia["_id"]},
                {"$set": update_data}
            )
            attendance_response = serialize_attendance({
                **asistencia,
                **update_data
            })

            return {
                "success": True,
                "verified": True,
                "tipo": "salida",
                "message": "Salida registrada correctamente" if validacion["valido"] else "Salida registrada fuera de rango",
                "user": {
                    "nombre": f"{user.get('name', '')} {user.get('lname', '')}".strip(),
                    "dni": user.get("dni"),
                    "cargo": user.get("position")
                },
                "validacion_ubicacion": {
                    "valido": validacion["valido"],
                    "distancia": validacion["distancia"],
                    "radio_permitido": float(sede["radio"]) if sede else None,
                    "omitida_por_remoto": validacion.get("omitida_por_remoto", False)
                },
                "validacion_facial": {
                    "similarity": similarity,
                    "threshold": threshold
                },
                # ✅ Información de cumplimiento completo
                "schedule_compliance": {
                    "is_late_entry": schedule_compliance['isLateEntry'],
                    "minutes_late_entry": schedule_compliance['minutesLateEntry'],
                    "is_early_departure": schedule_compliance['isEarlyDeparture'],
                    "minutes_early_departure": schedule_compliance['minutesEarlyDeparture']
                } if stored_expected_schedule else None,
                "horas_trabajadas": horas_trabajadas,
                "attendance": attendance_response,
                # ✅ Convertir a local
                "entrada": utc_to_local(asistencia["entrada"]).isoformat(),
                "salida": salida_local.isoformat(),
                "timestamp": salida_local.isoformat()
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error en marcar_asistencia: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en el servidor: {str(e)}"
        )
