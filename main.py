from utils.video_utils import read_video, save_video
from tracker.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance.estimator import SpeedAndDistance_Estimator
import cv2
import numpy as np

def main():
    video_frames = read_video('input_video/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    # Camera Movement Estimator 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                  stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Balll
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])


    # Speed and Distance Estimator
    speed_and_distance = SpeedAndDistance_Estimator()
    speed_and_distance.add_speed_and_distance_to_tracks(tracks)
    # Assign Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquistion  
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player=player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    #Output the video and annotate
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    #Draw camera movement
    output_video_frames= camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and Distance
    output_video_frames=speed_and_distance.draw_speed_and_distance(output_video_frames, tracks)

    print(len(output_video_frames))
    print(tracks['players'][0][3]['team_color'])
    print(tracks['players'][0][16]['team_color'])

    save_video(output_video_frames, 'output_video/output_video.mp4')

if __name__== '__main__':
    main()