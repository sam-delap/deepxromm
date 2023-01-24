<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html>
  <head>
    <title>Python: module samtools</title>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  </head>
  <body bgcolor="#f0f0f8">
    <table width="100%" cellspacing=0 cellpadding=2 border=0 summary="heading">
      <tr bgcolor="#7799ee">
        <td valign=bottom>&nbsp; <br>
          <font color="#ffffff" face="helvetica, arial">&nbsp; <br>
            <big>
              <big>
                <strong>samtools</strong>
              </big>
            </big>
          </font>
        </td>
        <td align=right valign=bottom>
          <font color="#ffffff" face="helvetica, arial">
            <a href=".">index</a>
            <br>
            <a href="file:c%3A%5Cusers%5Csjcde%5Cdocuments%5Cdeadromm-tools%5Csamtools.py">c:\users\sjcde\documents\deadromm-tools\samtools.py</a>
          </font>
        </td>
      </tr>
    </table>
    <p>
      <tt>A&nbsp;Complete&nbsp;Set&nbsp;of&nbsp;User-Friendly&nbsp;Tools&nbsp;for&nbsp;DeepLabCut-XMAlab&nbsp;marker&nbsp;tracking</tt>
    </p>
    <p>
    <table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
      <tr bgcolor="#aa55cc">
        <td colspan=3 valign=bottom>&nbsp; <br>
          <font color="#ffffff" face="helvetica, arial">
            <big>
              <strong>Modules</strong>
            </big>
          </font>
        </td>
      </tr>
      <tr>
        <td bgcolor="#aa55cc">
          <tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt>
        </td>
        <td>&nbsp;</td>
        <td width="100%">
          <table width="100%" summary="list">
            <tr>
              <td width="25%" valign=top>
                <a href="cv2.html">cv2</a>
                <br>
                <a href="deeplabcut.html">deeplabcut</a>
                <br>
                <a href="math.html">math</a>
                <br>
              </td>
              <td width="25%" valign=top>
                <a href="numpy.html">numpy</a>
                <br>
                <a href="os.html">os</a>
                <br>
                <a href="pandas.html">pandas</a>
                <br>
              </td>
              <td width="25%" valign=top>
                <a href="matplotlib.pyplot.html">matplotlib.pyplot</a>
                <br>
                <a href="warnings.html">warnings</a>
                <br>
                <a href="deeplabcut.utils.xrommtools.html">deeplabcut.utils.xrommtools</a>
                <br>
              </td>
              <td width="25%" valign=top></td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
    <p>
    <table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
      <tr bgcolor="#eeaa77">
        <td colspan=3 valign=bottom>&nbsp; <br>
          <font color="#ffffff" face="helvetica, arial">
            <big>
              <strong>Functions</strong>
            </big>
          </font>
        </td>
      </tr>
      <tr>
        <td bgcolor="#eeaa77">
          <tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt>
        </td>
        <td>&nbsp;</td>
        <td width="100%">
          <dl>
            <dt>
              <a name="-analyze_videos">
                <strong>analyze_videos</strong>
              </a>(working_dir='C:\\Users\\sjcde\\Documents\\deadROMM-tools')
            </dt>
            <dd>
              <tt>Analyze&nbsp;videos&nbsp;with&nbsp;a&nbsp;pre-existing&nbsp;network</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-autocorrect">
                <strong>autocorrect</strong>
              </a>(working_dir, search_area=15, threshold=8)
            </dt>
            <dd>
              <tt>Do&nbsp;XMAlab-style&nbsp;autocorrect&nbsp;on&nbsp;the&nbsp;tracked&nbsp;beads&nbsp;using&nbsp;contours</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-autocorrect_video">
                <strong>autocorrect_video</strong>
              </a>(cam, data_file, trial_name, working_dir, search_area, threshold)
            </dt>
            <dd>
              <tt>Perform&nbsp;autocorrect&nbsp;on&nbsp;a&nbsp;single&nbsp;video</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-create_new_project">
                <strong>create_new_project</strong>
              </a>(working_dir='C:\\Users\\sjcde\\Documents\\deadROMM-tools', experimenter='NA')
            </dt>
            <dd>
              <tt>Create&nbsp;a&nbsp;new&nbsp;xrommtools&nbsp;project</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-filter_image">
                <strong>filter_image</strong>
              </a>(image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.3)
            </dt>
            <dd>
              <tt>Filter&nbsp;the&nbsp;image&nbsp;to&nbsp;make&nbsp;it&nbsp;easier&nbsp;to&nbsp;see&nbsp;the&nbsp;bead</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-get_bodyparts_from_xma">
                <strong>get_bodyparts_from_xma</strong>
              </a>(path_to_trial)
            </dt>
            <dd>
              <tt>Pull&nbsp;the&nbsp;names&nbsp;of&nbsp;the&nbsp;XMAlab&nbsp;markers&nbsp;from&nbsp;the&nbsp;2Dpoints&nbsp;file</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-load_project">
                <strong>load_project</strong>
              </a>(working_dir='C:\\Users\\sjcde\\Documents\\deadROMM-tools', threshold=0.1)
            </dt>
            <dd>
              <tt>Load&nbsp;an&nbsp;existing&nbsp;project&nbsp;(only&nbsp;used&nbsp;internally/in&nbsp;testing)</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-show_crop">
                <strong>show_crop</strong>
              </a>(src, center, scale=5, contours=None, detected_marker=None)
            </dt>
            <dd>
              <tt>Display&nbsp;a&nbsp;visual&nbsp;of&nbsp;the&nbsp;marker&nbsp;and&nbsp;Python's&nbsp;projected&nbsp;center</tt>
            </dd>
          </dl>
          <dl>
            <dt>
              <a name="-train_network">
                <strong>train_network</strong>
              </a>(working_dir='C:\\Users\\sjcde\\Documents\\deadROMM-tools')
            </dt>
            <dd>
              <tt>Start&nbsp;training&nbsp;xrommtools-compatible&nbsp;data</tt>
            </dd>
          </dl>
        </td>
      </tr>
    </table>
  </body>
</html>
